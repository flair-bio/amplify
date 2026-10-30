"""Resolve benchmark roles without loading a model or tokenizing data."""

from __future__ import annotations

from datasets import DatasetDict

from modules.evaluate.src.dataset.dataloader import (
    DatasetSplitConfig,
    load_evaluation_dataset,
    load_raw_evaluation_dataset,
)


def resolve_evaluation_data(
    dataset_id: str,
    *,
    split: DatasetSplitConfig,
    seed: int,
    validation_fraction: float,
    dataset: DatasetDict | None = None,
    revision: str | None = None,
    subset: str | None = None,
) -> DatasetDict:
    """Supplied datasets already contain resolved roles, as in CV callers."""
    if dataset is not None:
        return dataset
    return load_evaluation_dataset(
        dataset_id,
        seed=seed,
        validation_fraction=validation_fraction,
        split=split,
        revision=revision,
        subset=subset,
    )


def resolve_external_data(
    dataset_id: str,
    split: str,
    *,
    revision: str | None = None,
    subset: str | None = None,
) -> DatasetDict:
    raw = load_raw_evaluation_dataset(dataset_id, revision=revision, subset=subset)
    if split not in raw:
        raise ValueError(
            f"External dataset '{dataset_id}' has no split '{split}'. "
            f"Available splits: {sorted(raw.keys())}."
        )
    return DatasetDict({"test": raw[split]})
