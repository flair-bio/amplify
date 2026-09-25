from collections.abc import Iterator, Sequence
from itertools import islice
from typing import Literal

import numpy as np
import random
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from modules.pretrain.src.dataset.collator import DataCollator
from modules.pretrain.src.dataset.dataset import (
    DatasetConfig,
    EpochPlan,
    build_epoch_plan,
    PreparedSource,
    prepare_sources,
)


class DataLoaderConfig(BaseModel):
    """Config for dataset loading and dataloader construction."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = Field(None, description="Token budget per batch.")
    batch_size: int | None = Field(None, description="Number of sequences per batch.")
    dataloader_num_workers: int = Field(
        0, description="Number of data-loading worker processes."
    )
    persistent_workers: bool = Field(
        True, description="If True, keep dataloader workers alive between epochs."
    )
    pin_memory: bool = Field(
        True,
        description="If True, copy tensors into pinned memory before returning them.",
    )
    prefetch_factor: int = Field(
        4, description="Number of batches to prefetch per worker."
    )
    in_order: bool = Field(
        True,
        description=(
            "If False, yield batches as soon as any worker finishes instead of "
            "in sampler order, so one slow batch doesn't stall the rest. Batch "
            "order then varies between runs, and mid-epoch resume may repeat "
            "or skip up to num_workers * prefetch_factor batches. Only applies "
            "with dataloader_num_workers > 0."
        ),
    )
    drop_last: bool = Field(
        True,
        description=(
            "If True, drop trailing batches so every rank gets the same "
            "number of batches per epoch. Negligible data loss in practice."
        ),
    )


def seed_worker(worker_id: int) -> None:
    """Seed each dataloader worker uniquely (masks/truncation differ per worker)."""
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_worker_seed_generator(
    seed: int,
    epoch: int,
    split: str = "train",
    process_index: int = 0,
) -> torch.Generator:
    """Build the RNG PyTorch uses to derive dataloader worker seeds.

    Pass as ``DataLoader(generator=...)``. Without it, workers are seeded
    from the *global* torch RNG at iterator-creation time, so masks depend
    on ambient RNG state rather than the configured seed, breaking resume
    reproducibility. Keying on ``(seed, epoch, split, process_index)``
    instead makes worker seeds a pure function of config -- exact at
    epoch-boundary resume, though not mid-epoch.
    """
    # SeedSequence (not a hand-rolled `a*seed + b*epoch + ...`) so each
    # component gets its own slot and can't alias another.
    split_tag = {"train": 0, "val": 1}.get(split)
    if split_tag is None:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'val'.")

    (state,) = np.random.SeedSequence(
        [seed, epoch, split_tag, process_index]
    ).generate_state(1, dtype=np.uint64)
    generator = torch.Generator()
    # torch.manual_seed takes a signed 64-bit value.
    generator.manual_seed(int(state >> 1))
    return generator


class TokenBatchSampler(Sampler[list[int]]):
    """Groups global row ids into batches under a max-token limit (in order, no GPU sharding).

    ``row_ids`` are indices into the persistent ``ConcatDataset`` (an
    ``EpochPlan.row_ids`` array) -- batches yield these global ids directly,
    since the dataset is no longer pre-filtered/shuffled to this epoch's rows.

    Batch boundaries are computed once, up front, via a vectorized NumPy
    prefix-sum + ``searchsorted`` scan rather than a per-row Python loop --
    significant at full-corpus scale (millions of rows). ``__iter__`` then
    lazily slices per batch instead of materializing every batch up front.
    """

    def __init__(
        self,
        row_ids: Sequence[int],
        sequence_lengths: Sequence[int],
        max_tokens: int,
        max_sequence_length: int | None = None,
    ) -> None:
        self._row_ids = np.asarray(row_ids)
        self._boundaries = self._pack_token_batch_boundaries(
            sequence_lengths, max_tokens, max_sequence_length
        )

    @staticmethod
    def _pack_token_batch_boundaries(
        sequence_lengths: Sequence[int],
        max_tokens: int,
        max_sequence_length: int | None = None,
    ) -> np.ndarray:
        """Greedy-pack under ``max_tokens``, returning batch boundary indices.

        Returns an array of length ``num_batches + 1``: indices ``k``/``k+1``
        mark batch ``k``'s ``[start, end)`` span into ``sequence_lengths``
        (and ``row_ids``). A sequence longer than ``max_tokens`` alone still
        gets its own one-item batch (``max(j, i + 1)`` below).

        ``sequence_lengths`` are raw residue counts; the collator truncates to
        ``max_sequence_length`` tokens (BOS/EOS included), so costs are capped
        there to avoid budgeting tokens that are never emitted.
        """
        costs = np.asarray(sequence_lengths, dtype=np.int64) + 2  # +2 for BOS/EOS
        if max_sequence_length is not None:
            np.minimum(costs, max_sequence_length, out=costs)
        n = len(costs)
        if n == 0:
            return np.zeros(1, dtype=np.int64)

        cumsum = np.empty(n + 1, dtype=np.int64)
        cumsum[0] = 0
        np.cumsum(costs, out=cumsum[1:])

        boundaries = [0]
        i = 0
        while i < n:
            # Largest j (> i) with cumsum[j] - cumsum[i] <= max_tokens: the
            # rightmost insertion point for `cumsum[i] + max_tokens` in the
            # monotonic cumsum array, minus one for the exclusive endpoint.
            j = int(np.searchsorted(cumsum, cumsum[i] + max_tokens, side="right")) - 1
            j = max(j, i + 1)  # always include at least one sequence per batch
            boundaries.append(j)
            i = j
        return np.array(boundaries, dtype=np.int64)

    def __iter__(self) -> Iterator[list[int]]:
        for k in range(len(self._boundaries) - 1):
            yield self._row_ids[self._boundaries[k] : self._boundaries[k + 1]].tolist()

    def __len__(self) -> int:
        return max(len(self._boundaries) - 1, 0)


class _TruncatedBatches(Sampler[list[int]]):
    """Lazily yields only the first ``count`` batches of ``batches``.

    Used by ``_drop_for_even_ranks`` so truncating for even-rank divisibility
    doesn't force materializing ``batches`` (which may be a lazy
    ``TokenBatchSampler``) just to slice it.
    """

    def __init__(
        self, batches: list[list[int]] | Sampler[list[int]], count: int
    ) -> None:
        self._batches = batches
        self._count = count

    def __iter__(self) -> Iterator[list[int]]:
        return islice(iter(self._batches), self._count)

    def __len__(self) -> int:
        return self._count


def _drop_for_even_ranks(
    batches: list[list[int]] | Sampler[list[int]], num_processes: int
) -> list[list[int]] | Sampler[list[int]]:
    """Truncate ``batches`` so its count divides evenly by ``num_processes``.

    Works for both a plain list and a lazy sampler (e.g.
    ``TokenBatchSampler``): truncation goes through ``_TruncatedBatches``,
    which slices lazily instead of forcing full materialization.
    """
    if num_processes <= 1:
        return batches
    total = len(batches)
    even_count = total - (total % num_processes)
    if even_count == total:
        return batches
    return _TruncatedBatches(batches, even_count)


def build_epoch_dataloader(
    concat_dataset: ConcatDataset,
    epoch_plan: EpochPlan,
    collator: DataCollator,
    dataloader_config: DataLoaderConfig,
    num_processes: int = 1,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Build a map-style DataLoader for one epoch, over the persistent ``concat_dataset``.

    ``concat_dataset`` is built once and reused unchanged across epochs;
    only ``epoch_plan`` varies. Pass ``num_processes`` so ``drop_last``
    gives every rank the same batch count, and ``generator`` (see
    ``build_worker_seed_generator``) so worker seeds don't depend on
    ambient RNG state.
    """
    if (dataloader_config.max_tokens is None) == (dataloader_config.batch_size is None):
        raise ValueError(
            "Specify exactly one of `dataloader.max_tokens` or `dataloader.batch_size`."
        )

    if dataloader_config.max_tokens is not None:
        batches: list[list[int]] | Sampler[list[int]] = TokenBatchSampler(
            epoch_plan.row_ids,
            epoch_plan.sequence_lengths,
            dataloader_config.max_tokens,
            collator.max_length,
        )
    else:
        row_ids = epoch_plan.row_ids
        batch_size = dataloader_config.batch_size
        assert batch_size is not None  # guaranteed by the XOR check above
        batches = [
            row_ids[i : i + batch_size].tolist()
            for i in range(0, len(row_ids), batch_size)
        ]

    if dataloader_config.drop_last:
        batches = _drop_for_even_ranks(batches, num_processes)

    return DataLoader(
        concat_dataset,
        batch_sampler=batches,
        collate_fn=collator,
        num_workers=dataloader_config.dataloader_num_workers,
        pin_memory=dataloader_config.pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(
            dataloader_config.persistent_workers
            if dataloader_config.dataloader_num_workers > 0
            else False
        ),
        prefetch_factor=(
            dataloader_config.prefetch_factor
            if dataloader_config.dataloader_num_workers > 0
            else None
        ),
        in_order=dataloader_config.in_order,
    )


def build_split_dataloader(
    prepared_sources: list[PreparedSource],
    concat_dataset: ConcatDataset,
    dataset_config: DatasetConfig,
    collator: DataCollator,
    dataloader_config: DataLoaderConfig,
    split: Literal["train", "val"],
    epoch: int = 0,
    num_processes: int = 1,
    process_index: int = 0,
) -> DataLoader:
    """Build a DataLoader for one split from already-prepared sources.

    ``concat_dataset`` must be the persistent ``ConcatDataset`` built once
    over ``prepared_sources`` in ``run_pretrain.py`` -- passing a fresh one
    each call would work but defeats the point of building it once.

    For ``split="val"``, leave ``epoch`` at its default (0): the val set is
    fixed-seed and must not vary per epoch. For ``split="train"``, pass the
    current epoch (or the resumed epoch, when resuming) since the train set
    is epoch-seeded.
    """

    # Enforce epoch=0 for validation to prevent dynamic seeding mistakes
    if split == "val" and epoch != 0:
        raise ValueError(
            f"Epoch must be 0 for split='val' to ensure a fixed evaluation set across epochs. "
            f"Got epoch={epoch} instead."
        )

    epoch_plan = build_epoch_plan(
        prepared_sources,
        epoch=epoch,
        seed=dataset_config.seed,
        dataset_cfg=dataset_config,
        split=split,
    )
    return build_epoch_dataloader(
        concat_dataset,
        epoch_plan,
        collator,
        dataloader_config,
        num_processes=num_processes,
        generator=build_worker_seed_generator(
            seed=dataset_config.seed,
            epoch=epoch,
            split=split,
            process_index=process_index,
        ),
    )
