"""Cross-validation fold planning over raw (untokenized) evaluation data.

Builds a pool from configured source roles (never the official test split),
assigns outer folds plus a group/stratum-aware inner validation holdout, and
validates partitions before any model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict, concatenate_datasets
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from modules.evaluate.src.dataset.dataloader import DatasetSplitConfig

CVStrategy = Literal[
    "kfold", "stratified_kfold", "group_kfold", "stratified_group_kfold"
]
CV_TASK_TYPES = frozenset({"sequence_classification", "sequence_regression"})
ROLES = ("train", "validation", "test")


class CrossValidationConfig(BaseModel):
    """Outer K-fold evaluation with an inner validation holdout per fold."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_splits: int = Field(5, ge=2)
    strategy: CVStrategy = "stratified_kfold"
    group_column: str | None = None
    pool_roles: list[Literal["train", "validation"]] = Field(
        default_factory=lambda: ["train", "validation"]
    )
    split_seed: int = 710019
    # Held out as one fold of an inner round(1 / fraction)-fold split using the
    # same strategy, so groups and strata are respected.
    inner_validation_fraction: float = Field(0.1, gt=0.0, le=0.5)

    @field_validator("pool_roles")
    @classmethod
    def _validate_pool_roles(cls, roles: list[str]) -> list[str]:
        if not roles or len(roles) != len(set(roles)):
            raise ValueError(
                "cross_validation.pool_roles must be distinct and nonempty."
            )
        return roles

    @model_validator(mode="after")
    def _validate_grouping(self) -> CrossValidationConfig:
        if self.grouped != (self.group_column is not None):
            raise ValueError(
                "cross_validation.group_column is required for, and only valid "
                "with, group strategies."
            )
        return self

    @property
    def grouped(self) -> bool:
        return "group" in self.strategy

    @property
    def stratified(self) -> bool:
        return self.strategy.startswith("stratified")

    @property
    def inner_n_splits(self) -> int:
        return max(2, round(1 / self.inner_validation_fraction))


@dataclass(frozen=True)
class Fold:
    """Pool row indices for one outer fold's train/validation/test roles."""

    fold_id: int
    train: list[int]
    validation: list[int]
    test: list[int]

    def roles(self) -> dict[str, list[int]]:
        return {role: getattr(self, role) for role in ROLES}

    def dataset(self, pool: Dataset) -> DatasetDict:
        """Materialize roles; ``test`` is the outer held-out fold."""
        return DatasetDict(
            {role: pool.select(indices) for role, indices in self.roles().items()}
        )


@dataclass(frozen=True)
class FoldPlan:
    pool: Dataset
    folds: list[Fold]
    assignments: pl.DataFrame
    fingerprint: str

    @property
    def cv_id(self) -> str:
        return self.fingerprint[:12]


def _splitter(strategy: CVStrategy, n_splits: int, seed: int) -> Any:
    if strategy == "group_kfold":
        return GroupKFold(n_splits, shuffle=True, random_state=seed)
    splitter = {
        "kfold": KFold,
        "stratified_kfold": StratifiedKFold,
        "stratified_group_kfold": StratifiedGroupKFold,
    }[strategy]
    return splitter(n_splits, shuffle=True, random_state=seed)


def _split(
    config: CrossValidationConfig,
    n_splits: int,
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
) -> list[tuple[list[int], list[int]]]:
    """Split *indices* into (kept, held-out) pool indices for each fold."""
    splits = _splitter(config.strategy, n_splits, config.split_seed).split(
        np.zeros(len(indices)),
        labels[indices],
        None if groups is None else groups[indices],
    )
    return [
        (sorted(indices[kept].tolist()), sorted(indices[held].tolist()))
        for kept, held in splits
    ]


def _build_pool(
    dataset: DatasetDict,
    config: CrossValidationConfig,
    split_config: DatasetSplitConfig,
    columns: list[str],
    id_column: str,
) -> tuple[Dataset, pl.DataFrame]:
    """Concatenate pool roles with namespaced IDs; return the pool and provenance."""
    for role in config.pool_roles:
        if getattr(split_config, role) == split_config.test:
            raise ValueError(
                "Cross-validation pools must exclude the official test split."
            )
    parts, provenance = [], []
    for role in config.pool_roles:
        source = getattr(split_config, role)
        if source not in dataset:
            raise ValueError(
                f"Cross-validation pool role '{role}' (source split '{source}') "
                "must exist in the dataset."
            )
        part = dataset[source]
        missing = set(columns) - set(part.column_names)
        if missing:
            raise ValueError(
                f"Source split '{source}' lacks columns {sorted(missing)}."
            )
        source_ids = (
            part[id_column] if id_column in part.column_names else [None] * len(part)
        )
        ids = [f"{source}:{index}" for index in range(len(part))]
        parts.append(part.select_columns(columns).add_column(id_column, ids))
        provenance.append(
            pl.DataFrame(
                {
                    "id": ids,
                    "source_split": [source] * len(part),
                    "source_index": list(range(len(part))),
                    # JSON preserves the original ID type and missing values.
                    "source_id": [json.dumps(value) for value in source_ids],
                }
            )
        )
    return concatenate_datasets(parts), pl.concat(provenance)


def _validate_pool(
    config: CrossValidationConfig,
    sequences: list[str],
    labels: list[Any],
    groups: list[Any] | None,
    classification: bool,
) -> None:
    for label in labels:
        if (
            isinstance(label, bool)
            or not isinstance(label, (int, float))
            or not math.isfinite(label)
        ):
            raise ValueError(
                f"Cross-validation labels must be finite scalars, got {label!r}."
            )
    if groups is not None and any(group is None for group in groups):
        raise ValueError(f"Group column '{config.group_column}' must be non-null.")
    # Identical sequences must not straddle folds, so they need a shared group.
    by_sequence: dict[str, set[Any]] = defaultdict(set)
    for index, sequence in enumerate(sequences):
        by_sequence[sequence].add(None if groups is None else groups[index])
    duplicates = Counter(sequences)
    if any(
        count > 1 and (groups is None or len(by_sequence[sequence]) > 1)
        for sequence, count in duplicates.items()
    ):
        raise ValueError(
            "Duplicate sequences found in the cross-validation pool; use a group "
            "strategy whose group column assigns duplicates the same group."
        )
    n_units = len(set(groups)) if groups is not None else len(labels)
    if n_units < config.n_splits:
        raise ValueError(
            f"{n_units} groups/examples cannot form {config.n_splits} folds."
        )
    if classification and config.stratified:
        rare = {k: v for k, v in Counter(labels).items() if v < config.n_splits}
        if rare:
            raise ValueError(
                f"Stratified CV needs at least n_splits={config.n_splits} "
                f"examples per class; got {rare}."
            )


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def make_fold_plan(
    dataset: DatasetDict,
    config: CrossValidationConfig,
    split_config: DatasetSplitConfig,
    *,
    task_type: str,
    sequence_column: str,
    label_column: str,
    id_column: str,
) -> FoldPlan:
    """Plan and validate outer folds with inner holdouts over the CV pool."""
    if task_type not in CV_TASK_TYPES:
        raise ValueError(
            f"Cross-validation supports task types {sorted(CV_TASK_TYPES)}, "
            f"got '{task_type}'."
        )
    classification = task_type == "sequence_classification"
    columns = [sequence_column, label_column]
    if config.group_column is not None:
        columns.append(config.group_column)
    pool, provenance = _build_pool(dataset, config, split_config, columns, id_column)

    sequences, label_list = pool[sequence_column], pool[label_column]
    group_list = pool[config.group_column] if config.group_column else None
    _validate_pool(config, sequences, label_list, group_list, classification)
    labels = np.asarray(label_list)
    groups = np.asarray(group_list) if group_list is not None else None
    classes = set(label_list)

    folds = []
    for fold_id, (outer_train, test) in enumerate(
        _split(config, config.n_splits, np.arange(len(pool)), labels, groups)
    ):
        train, validation = _split(
            config, config.inner_n_splits, np.asarray(outer_train), labels, groups
        )[0]
        fold = Fold(fold_id, train, validation, test)
        for role, indices in fold.roles().items():
            if not indices:
                raise ValueError(f"Fold {fold_id} has an empty {role} role.")
            if classification and role != "test":
                missing = classes - set(labels[indices].tolist())
                if missing:
                    raise ValueError(
                        f"Fold {fold_id} {role} role is missing classes "
                        f"{sorted(missing)}; adjust grouping, n_splits, or "
                        "inner_validation_fraction."
                    )
        folds.append(fold)

    group_json: list[str | None] = (
        [json.dumps(group) for group in group_list]
        if group_list is not None
        else [None] * len(pool)
    )
    assignments = pl.concat(
        provenance.with_columns(
            pl.Series("group", group_json, dtype=pl.String),
            pl.lit(fold.fold_id).alias("fold_id"),
            pl.Series("role", _role_column(fold, len(pool))),
        )
        for fold in folds
    )
    fingerprint = _fingerprint(
        {
            "config": config.model_dump(mode="json", exclude={"enabled"}),
            "split": split_config.model_dump(mode="json"),
            "task_type": task_type,
            "data": [pool[id_column], sequences, label_list, group_list],
            "roles": assignments["role"].to_list(),
        }
    )
    return FoldPlan(pool, folds, assignments, fingerprint)


def _role_column(fold: Fold, n_rows: int) -> list[str]:
    roles = [""] * n_rows
    for role, indices in fold.roles().items():
        for index in indices:
            roles[index] = role
    return roles
