"""Baseline observations, identities, loading, validation and row alignment."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import polars as pl
from datasets import Dataset, DatasetDict

from modules.evaluate.src.contracts.config import (
    BaselineName,
    BaselinesConfig,
    DatasetWorkspaceConfig,
)


def identity_hash(payload: Any) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(payload):
        digest.update(chunk.encode())
    return digest.hexdigest()


def sequence_rows(
    dataset: Dataset,
    *,
    sequence_column: str,
    label_column: str,
    id_column: str,
    task_type: str,
    num_labels: int | None,
) -> list[dict[str, Any]]:
    """Preserve source observations, including repeated IDs."""
    if not len(dataset):
        raise ValueError("A shared baseline requires nonempty train and test data.")
    rows = []
    columns = [sequence_column, label_column]
    if id_column in dataset.column_names:
        columns.append(id_column)
    for index, row in enumerate(dataset.select_columns(columns)):
        label = row[label_column]
        if not isinstance(label, (int, float)) or not math.isfinite(label):
            raise ValueError("Sequence baseline targets must be finite scalars.")
        if task_type == "sequence_classification":
            if label != int(label) or not 0 <= label < (num_labels or 0):
                raise ValueError("Baseline class indices are outside num_labels.")
            label = int(label)
        else:
            label = float(label)
        identifier = row.get(id_column)
        rows.append(
            {
                "id": index if identifier is None else identifier,
                "sequence": row[sequence_column],
                "label": label,
            }
        )
    return rows


def baseline_context(
    workspace: DatasetWorkspaceConfig,
    dataset: DatasetDict,
    names: list[BaselineName],
    *,
    sequence_column: str,
    label_column: str,
    id_column: str,
    split_roles: dict[str, Any],
) -> dict[str, Any]:
    """Describe expected artifacts before model-specific preparation."""
    BaselinesConfig(names=names).validate_task(workspace.task_type)
    arguments: dict[str, Any] = {
        "sequence_column": sequence_column,
        "label_column": label_column,
        "id_column": id_column,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
    }
    train_rows = sequence_rows(dataset["train"], **arguments)
    test_rows = sequence_rows(dataset["test"], **arguments)
    fit = {
        "version": 1,
        "dataset": workspace.dataset_repo_id or workspace.dataset_name,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
        "fit_policy": "train_only",
        "split_roles": {"name": split_roles["name"], "train": split_roles["train"]},
        "columns": [sequence_column, label_column, id_column],
        "train_fingerprint": identity_hash(train_rows),
    }
    return {
        "fit": fit,
        "test": {
            "dataset": workspace.dataset_repo_id or workspace.dataset_name,
            "split": split_roles["test"],
            "fingerprint": identity_hash(test_rows),
        },
        "root": str((workspace.dataset_root / "baselines").resolve()),
        "names": names,
    }


def external_baseline_context(
    context: dict[str, Any],
    workspace: DatasetWorkspaceConfig,
    dataset: DatasetDict,
    *,
    dataset_id: str,
    split: str,
    sequence_column: str,
    label_column: str,
    id_column: str,
) -> dict[str, Any]:
    rows = sequence_rows(
        dataset["test"],
        sequence_column=sequence_column,
        label_column=label_column,
        id_column=id_column,
        task_type=workspace.task_type,
        num_labels=workspace.num_labels,
    )
    return {
        **context,
        "test": {
            "dataset": dataset_id,
            "split": split,
            "fingerprint": identity_hash(rows),
        },
    }


def artifact_identity(context: dict[str, Any], name: str) -> dict[str, Any]:
    return {"baseline": name, "fit": context["fit"], "test": context["test"]}


def artifact_directory(context: dict[str, Any], name: str) -> Path:
    identity = artifact_identity(context, name)
    return (
        Path(context["root"])
        / name
        / f"fit-{identity_hash(identity['fit'])[:12]}"
        / f"test-{identity_hash(identity['test'])[:12]}"
    )


def load_baseline_artifact(
    context: dict[str, Any], name: str
) -> tuple[pl.DataFrame, Path]:
    directory = artifact_directory(context, name)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared baseline '{name}' at {directory}. "
            "Run run_evaluate.py --baselines with the same dataset/task "
            "configuration before scoring."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != artifact_identity(context, name):
        raise ValueError(f"Incompatible baseline manifest: {manifest_path}")
    path = directory / "predictions.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"Corrupt baseline predictions: {path}")
    frame = pl.read_parquet(path)
    required = {"id", "label", "prediction"}
    if name == "majority_class":
        required.add("probability")
    if not required.issubset(frame.columns) or frame.height != manifest["rows"]:
        raise ValueError(f"Invalid baseline prediction schema/count: {path}")
    return frame, path


def align_baseline(
    baseline: pl.DataFrame, predictions: pl.DataFrame, *, regression: bool
) -> pl.DataFrame:
    """Align repeated observations without multiplying rows in an ID join."""

    def row_key(identifier: Any, label: Any) -> str:
        # Prediction parquet and torch regression labels use float32.
        if regression:
            label = struct.unpack("f", struct.pack("f", float(label)))[0]
        return json.dumps([identifier, label], allow_nan=False)

    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, row in enumerate(baseline.select("id", "label").iter_rows()):
        positions[row_key(*row)].append(index)
    indices = []
    for row in predictions.select("id", "label").iter_rows():
        candidates = positions[row_key(*row)]
        if not candidates:
            raise ValueError(
                "Shared baseline IDs/labels do not match model observations."
            )
        indices.append(candidates.popleft())
    if any(positions.values()):
        raise ValueError("Shared baseline and model have different observation counts.")
    return baseline[indices].drop("label")
