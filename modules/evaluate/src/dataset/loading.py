"""Dataset source loading and deterministic split resolution."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset  # type: ignore[attr-defined]

from modules.evaluate.src.config import DEFAULT_SPLIT_NAME, DatasetSplitConfig
from modules.evaluate.src.dataset.contact import prepare_contact_dataset

logger = logging.getLogger(__name__)
_DEFAULT_VALIDATION_FRACTION = 0.1


def ensure_validation_split(
    dataset: DatasetDict,
    seed: int = 710019,
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION,
) -> DatasetDict:
    """Carve a ``validation`` split out of ``train`` if one isn't already present.

    Some evaluation datasets only ship ``train``/``test``. Downstream steps
    (early stopping, hyperparameter-search scoring) require a ``validation``
    split to function, so rather than erroring out we synthesize one here via
    a plain seeded random split of ``train``. Leaves *dataset* untouched if it
    already has a ``validation`` split, or if it has no ``train`` split to
    split from.
    """
    if "validation" in dataset or "train" not in dataset:
        return dataset

    split = dataset["train"].train_test_split(test_size=validation_fraction, seed=seed)
    logger.warning(
        "Dataset has no 'validation' split; synthesized one from %.0f%% of "
        "'train' (seed=%d, %d train / %d validation rows remaining).",
        validation_fraction * 100,
        seed,
        len(split["train"]),
        len(split["test"]),
    )
    new_dataset = DatasetDict({**dataset})
    new_dataset["train"] = split["train"]
    new_dataset["validation"] = split["test"]
    return new_dataset


def select_dataset_splits(
    dataset: DatasetDict, config: DatasetSplitConfig
) -> DatasetDict:
    """Select pipeline roles and explicitly requested final-refit-only splits."""
    selected = DatasetDict()
    for role, source in (
        ("train", config.train),
        ("validation", config.validation),
        ("test", config.test),
    ):
        if source is None:
            continue
        if source in dataset:
            selected[role] = dataset[source]
            continue
        # Preserve existing support for datasets without a canonical role.
        # A missing validation role can also be synthesized from train below.
        if source == role or role == "validation":
            continue
        raise ValueError(
            f"Configured {role} source split '{source}' was not found. "
            f"Available splits: {sorted(dataset.keys())}."
        )
    for source in config.refit_additional_splits:
        if source not in dataset:
            raise ValueError(
                "Configured final-refit source split "
                f"'{source}' was not found. Available splits: {sorted(dataset.keys())}."
            )
        selected[source] = dataset[source]
    return selected


def load_raw_evaluation_dataset(
    dataset_repo_id: str | Path,
    *,
    config_name: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> DatasetDict:
    """Load one dataset configuration, preserving all of its source splits."""
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    path = Path(dataset_repo_id)
    files: dict[str, list[str]] = {}
    if path.is_dir():
        if revision is not None:
            raise ValueError("revision applies only to Hub datasets")
        # A standalone artifact can still be passed with its suite config name.
        # Suite roots instead use the root card's explicit configuration mapping.
        for file in sorted((path / "data").glob("*.parquet")):
            match = re.fullmatch(r"(.+)-\d+-of-\d+\.parquet", file.name)
            if match is not None:
                files.setdefault(match[1], []).append(str(file))
    if files:
        dataset = load_dataset("parquet", data_files=files, **kwargs)
    else:
        if not path.exists() and (
            isinstance(dataset_repo_id, Path) or path.is_absolute()
        ):
            raise FileNotFoundError(path)
        if revision is not None:
            kwargs["revision"] = revision
        if config_name is not None:
            kwargs["name"] = config_name
        dataset = load_dataset(str(dataset_repo_id), **kwargs)
    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict with train/validation/test splits from "
            f"'{dataset_repo_id}', got {type(dataset)}."
        )
    return dataset


def load_evaluation_dataset(
    dataset_repo_id: str,
    seed: int = 710019,
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION,
    split: DatasetSplitConfig | None = None,
    *,
    config_name: str | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Load the configured evaluation and final-refit splits locally or from the Hub.

    If the loaded dataset has no ``validation`` split, one is synthesized
    from ``train`` (see :func:`ensure_validation_split`).
    """
    kwargs: dict[str, Any] = {}
    if config_name is not None:
        kwargs["config_name"] = config_name
    if revision is not None:
        kwargs["revision"] = revision
    dataset = load_raw_evaluation_dataset(dataset_repo_id, **kwargs)
    if any("positive_pairs" in data.column_names for data in dataset.values()):
        if not Path(dataset_repo_id).is_dir() and (
            revision is None or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
        ):
            raise ValueError(
                "Hub contact data requires a pinned 40-character commit SHA"
            )
        if split is not None and split != DatasetSplitConfig(name=DEFAULT_SPLIT_NAME):
            raise ValueError(
                "CASP contacts require unchanged fixed train/validation/test splits"
            )
        prepare_contact_dataset(dataset, source=dataset_repo_id, revision=revision)
        return dataset
    if split is not None:
        dataset = select_dataset_splits(dataset, split)
    return ensure_validation_split(
        dataset, seed=seed, validation_fraction=validation_fraction
    )


def resolve_evaluation_data(
    dataset_id: str,
    *,
    split: DatasetSplitConfig,
    seed: int,
    validation_fraction: float,
    dataset: DatasetDict | None = None,
    config_name: str | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Load benchmark splits, or use supplied data with already resolved roles."""
    if dataset is not None:
        return dataset
    return load_evaluation_dataset(
        dataset_id,
        seed=seed,
        validation_fraction=validation_fraction,
        split=split,
        config_name=config_name,
        revision=revision,
    )


def resolve_external_data(
    dataset_id: str,
    split: str,
    *,
    config_name: str | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Expose one validated source split as an external test dataset."""
    kwargs: dict[str, Any] = {}
    if config_name is not None:
        kwargs["config_name"] = config_name
    if revision is not None:
        kwargs["revision"] = revision
    raw = load_raw_evaluation_dataset(dataset_id, **kwargs)
    if split not in raw:
        raise ValueError(
            f"External dataset '{dataset_id}' has no split '{split}'. "
            f"Available splits: {sorted(raw.keys())}."
        )
    return DatasetDict({"test": raw[split]})
