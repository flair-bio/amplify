"""
Load evaluation data, tokenize it once, and construct the split-specific
DataLoaders used by the evaluation step.

The workflow is intentionally eager: evaluation datasets are small enough to
fit in memory, so we load the full Hugging Face ``DatasetDict`` first, then
tokenize every split a single time, and finally hand the pre-tokenized splits
to PyTorch ``DataLoader`` instances.

Tokenization defaults to four processes and dataloading to the main process, tuned
independently. Dataloader worker counts apply per rank and per loader; choose
them to fit the job's CPU allocation.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from typing import Any

import polars as pl
import torch
from datasets import Dataset, DatasetDict  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Sampler

from modules.evaluate.src.dataset.collator import EvaluationCollator

logger = logging.getLogger(__name__)


def metric_groups_by_id(
    dataset: Dataset, group_column: str | None, id_column: str = "id"
) -> dict[str, Any]:
    """Retain metric groups by example ID before tokenization drops source metadata."""
    if group_column is None:
        return {}
    if group_column not in dataset.column_names:
        raise ValueError(
            f"Grouped-metric data must contain metric group column '{group_column}'."
        )
    if id_column not in dataset.column_names:
        raise ValueError(f"Grouped metrics require the source ID column '{id_column}'.")
    groups = {}
    for identifier, group in zip(
        dataset[id_column], dataset[group_column], strict=True
    ):
        key = json.dumps(identifier, allow_nan=False)
        if identifier is None or key in groups:
            raise ValueError("Grouped metrics require non-null, unique original IDs.")
        if group is None or (isinstance(group, float) and not math.isfinite(group)):
            raise ValueError("Metric group IDs must be non-null and finite.")
        groups[key] = group
    return groups


def metric_groups_for_ids(
    ids: list[Any], groups_by_id: dict[str, Any], split_name: str
) -> list[Any]:
    """Resolve grouped metrics in the same order as inferred examples."""
    keys = [json.dumps(identifier, allow_nan=False) for identifier in ids]
    if len(set(keys)) != len(keys) or set(keys) != set(groups_by_id):
        raise ValueError(f"Grouped predictions do not match prepared {split_name} IDs.")
    return [groups_by_id[key] for key in keys]


def attach_metric_groups(
    predictions: pl.DataFrame,
    groups_by_id: dict[str, Any],
    group_column: str | None,
) -> pl.DataFrame:
    """Attach source metric groups to predictions in their inference order."""
    if group_column is None:
        return predictions
    return predictions.with_columns(
        pl.Series(
            group_column,
            metric_groups_for_ids(predictions["id"].to_list(), groups_by_id, "test"),
        )
    )


# Default mega-batch sizing for LengthGroupedSampler, mirroring
# transformers.trainer_pt_utils.get_length_grouped_indices.
_MEGA_BATCH_MULT_BATCH_SIZE_MULTIPLIER = 4
_MEGA_BATCH_MULT_CAP = 50

# Default share of `train` held out to synthesize a `validation` split when
# the dataset doesn't already have one.
_DEFAULT_VALIDATION_FRACTION = 0.1


class DataLoaderConfig(BaseModel):
    """Config. for HF dataset loading and DataLoader construction."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(8, description="Number of examples per batch.")
    max_tokens_per_batch: int | None = Field(
        None, gt=0, description="Packed token budget per batch."
    )
    num_workers: int = Field(
        0,
        ge=0,
        description=(
            "Number of dataloader worker processes per rank and per loader. "
            "Defaults to 0 to avoid concurrent per-split worker pools."
        ),
    )
    pin_memory: bool = Field(
        True, description="Copy tensors into pinned memory before returning them."
    )
    persistent_workers: bool = Field(
        False, description="Keep dataloader workers alive between epochs."
    )
    seed: int = Field(
        710019,
        description=(
            "Seed for the train DataLoader's shuffling generator, so batch "
            "order is reproducible independent of global RNG state."
        ),
    )
    length_grouped_sampling: bool = Field(
        False,
        description=(
            "Batch examples of similar tokenized length together (via "
            "LengthGroupedSampler) instead of pure random shuffling. Reduces "
            "the amount of padding the collator has to add per batch, at the "
            "cost of batches no longer being uniformly randomly composed."
        ),
    )


def _get_length_grouped_indices(
    lengths: list[int],
    batch_size: int,
    mega_batch_mult: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Return dataset indices ordered so same-length examples end up in the same batch.

    Mirrors ``transformers.trainer_pt_utils.get_length_grouped_indices``:
    indices are shuffled, chunked into "mega-batches" of
    ``batch_size * mega_batch_mult`` examples, each mega-batch is sorted by
    length (descending), and the mega-batches are then concatenated back
    together. The longest example overall is swapped into the very first
    batch, which surfaces any out-of-memory error immediately rather than
    partway through training.
    """
    if mega_batch_mult is None:
        # Default straight from the HF implementation: big enough mega-batches
        # to meaningfully sort by length, capped at 50 to bound the cost of
        # shuffling within each chunk.
        mega_batch_mult = min(
            len(lengths) // (batch_size * _MEGA_BATCH_MULT_BATCH_SIZE_MULTIPLIER),
            _MEGA_BATCH_MULT_CAP,
        )
        if not mega_batch_mult:
            # Dataset is smaller than batch_size * 4: sort the whole dataset in
            # one mega-batch so items with similar lengths end up in the same
            # batch (instead of falling back to mega_batch_mult=1, which would
            # make the mega-batch the same size as a regular batch and not
            # provide any meaningful grouping).
            mega_batch_mult = max(1, -(-len(lengths) // batch_size))
    mega_batch_size = mega_batch_mult * batch_size

    indices = torch.randperm(len(lengths), generator=generator).tolist()
    megabatches = [
        indices[i : i + mega_batch_size]
        for i in range(0, len(lengths), mega_batch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]

    # Make sure the longest example is in the first batch so an OOM surfaces
    # immediately, rather than after training has been running for a while.
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = megabatch_maximums.index(max(megabatch_maximums))
    megabatches[0][0], megabatches[max_idx][0] = (
        megabatches[max_idx][0],
        megabatches[0][0],
    )

    return [index for megabatch in megabatches for index in megabatch]


class LengthGroupedSampler(Sampler[int]):
    """Yields dataset indices grouped by (tokenized) sequence length.

    Batches sampled consecutively from the returned index order share similar
    lengths, so the collator pads less per batch than it would with a plain
    random shuffle for training. Evaluation sorts by length deterministically
    so repeated inference passes keep predictions aligned by example.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        generator: torch.Generator | None = None,
        shuffle: bool = True,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(
                sorted(
                    range(len(self.lengths)),
                    key=lambda index: self.lengths[index],
                    reverse=True,
                )
            )
        indices = _get_length_grouped_indices(
            self.lengths, self.batch_size, generator=self.generator
        )
        return iter(indices)


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Greedily group examples into batches whose summed lengths stay within ``max_tokens``.

    Batch membership is fixed at construction (an example longer than the
    budget gets its own batch); with ``shuffle`` only the batch order changes
    between epochs.
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool,
        generator: torch.Generator,
    ) -> None:
        self.shuffle = shuffle
        self.generator = generator
        indices = (
            torch.randperm(len(lengths), generator=generator).tolist()
            if shuffle
            else list(range(len(lengths)))
        )
        self.batches: list[list[int]] = []
        current: list[int] = []
        cost = 0
        for index in indices:
            if current and cost + lengths[index] > max_tokens:
                self.batches.append(current)
                current = []
                cost = 0
            current.append(index)
            cost += lengths[index]
        if current:
            self.batches.append(current)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        order = (
            torch.randperm(len(self.batches), generator=self.generator).tolist()
            if self.shuffle
            else range(len(self.batches))
        )
        return (self.batches[index] for index in order)


def token_lengths(dataset: Dataset) -> list[int]:
    """Read token-list lengths without decoding individual token IDs."""
    return dataset.with_format("polars")["input_ids"].list.len().to_list()


def build_dataloaders(
    dataset: DatasetDict,
    collator: EvaluationCollator,
    config: DataLoaderConfig,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    """Build train/validation/test :class:`DataLoader` instances from a loaded dataset.

    Any split absent from *dataset* yields ``None`` for that DataLoader.
    """
    common_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
    )

    def _build(
        split: str, shuffle: bool, generator: torch.Generator | None = None
    ) -> DataLoader | None:
        if split not in dataset:
            return None
        if config.max_tokens_per_batch is not None:
            assert generator is not None
            lengths = token_lengths(dataset[split])
            sampler = TokenBudgetBatchSampler(
                lengths, config.max_tokens_per_batch, shuffle, generator
            )
            return DataLoader(
                dataset[split],
                batch_sampler=sampler,
                **{
                    key: value
                    for key, value in common_kwargs.items()
                    if key != "batch_size"
                },
            )
        if config.length_grouped_sampling:
            lengths = token_lengths(dataset[split])
            sampler = LengthGroupedSampler(
                lengths, config.batch_size, generator=generator, shuffle=shuffle
            )
            return DataLoader(dataset[split], sampler=sampler, **common_kwargs)
        return DataLoader(
            dataset[split], shuffle=shuffle, generator=generator, **common_kwargs
        )

    # Explicit, per-split generators (rather than global torch RNG state) so
    # every split's batch order is reproducible from ``config.seed`` alone.
    train_dataloader = _build(
        "train", shuffle=True, generator=torch.Generator().manual_seed(config.seed)
    )
    val_dataloader = _build(
        "validation",
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    test_dataloader = _build(
        "test", shuffle=False, generator=torch.Generator().manual_seed(config.seed)
    )
    return train_dataloader, val_dataloader, test_dataloader
