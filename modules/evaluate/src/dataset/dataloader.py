"""
Load evaluation data, tokenize it once, and construct the split-specific
DataLoaders used by the evaluation step.

The workflow is intentionally eager: evaluation datasets are small enough to
fit in memory, so we load the full Hugging Face ``DatasetDict`` first, then
tokenize every split a single time, and finally hand the pre-tokenized splits
to PyTorch ``DataLoader`` instances.

The worker defaults reflect that pipeline: when the user does not specify a
worker count, we scale to available hardware with
``min(os.cpu_count(), num_gpus * 4)`` on GPU runs, and a small CPU-friendly
default on CPU-only runs. That keeps tokenization and batch assembly parallel
without oversubscribing the machine.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Iterator

import torch
from datasets import DatasetDict, load_dataset  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.utils.data import DataLoader, Sampler
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.dataset.collator import EvaluationCollator
from modules.evaluate.src.tasks import TaskType, get_task

logger = logging.getLogger(__name__)

# Default mega-batch sizing for LengthGroupedSampler, mirroring
# transformers.trainer_pt_utils.get_length_grouped_indices.
_MEGA_BATCH_MULT_BATCH_SIZE_MULTIPLIER = 4
_MEGA_BATCH_MULT_CAP = 50

# Default share of `train` held out to synthesize a `validation` split when
# the dataset doesn't already have one.
_DEFAULT_VALIDATION_FRACTION = 0.1


DEFAULT_SPLIT_NAME = "default"


class DatasetSplitConfig(BaseModel):
    """Map dataset-specific split names onto evaluation pipeline roles.

    With only ``name`` set, roles resolve to ``<role>_<name>`` (``train``,
    ``validation``, ``test`` for the ``default`` name). Set a role explicitly
    to point at a split that doesn't follow that convention.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        DEFAULT_SPLIT_NAME,
        min_length=1,
        description="Name recorded in prediction and score metadata.",
    )
    train: str = "train"
    validation: str | None = "validation"
    test: str = "test"

    @model_validator(mode="after")
    def _derive_split_names(self) -> DatasetSplitConfig:
        """Suffix each role not set explicitly with a non-default split name."""
        if self.name == DEFAULT_SPLIT_NAME:
            return self
        for role in ("train", "validation", "test"):
            if role not in self.model_fields_set:
                setattr(self, role, f"{role}_{self.name}")
        return self


class DataLoaderConfig(BaseModel):
    """Config. for HF dataset loading and DataLoader construction."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(16, description="Number of examples per batch.")
    max_tokens_per_batch: int | None = Field(
        None, gt=0, description="Packed token budget per batch."
    )
    num_workers: int | None = Field(
        None,
        description=(
            "Number of dataloader worker processes. Defaults to "
            "min(os.cpu_count(), num_gpus * 4) on GPU runs and up to 4 "
            "workers on CPU-only runs when not set."
        ),
    )
    pin_memory: bool = Field(
        True, description="Copy tensors into pinned memory before returning them."
    )
    persistent_workers: bool = Field(
        True, description="Keep dataloader workers alive between epochs."
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
    random shuffle. Order is otherwise randomized every ``__iter__`` call
    (mega-batches are reshuffled), so this is a drop-in replacement for
    ``shuffle=True`` rather than a fully sorted/deterministic ordering.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.generator = generator

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        indices = _get_length_grouped_indices(
            self.lengths, self.batch_size, generator=self.generator
        )
        return iter(indices)


class TokenBudgetBatchSampler(Sampler[list[int]]):
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


def default_num_workers() -> int:
    """Return a hardware-aware default worker count.

    Uses ``min(os.cpu_count(), num_gpus * 4)`` when GPUs are present, and a
    conservative CPU-only default (up to 4 workers) otherwise.
    """
    num_gpus = torch.cuda.device_count()
    cpu_count = os.cpu_count() or 1
    if num_gpus == 0:
        return min(cpu_count, 4)
    return max(0, min(cpu_count, num_gpus * 4))


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
    """Select source splits and expose them as train/validation/test."""
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
    return selected


def load_evaluation_dataset(
    dataset_repo_id: str,
    seed: int = 710019,
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION,
    split: DatasetSplitConfig | None = None,
) -> DatasetDict:
    """Load the full train/validation/test splits from the Hugging Face Hub.

    If the loaded dataset has no ``validation`` split, one is synthesized
    from ``train`` (see :func:`ensure_validation_split`).
    """
    dataset = load_dataset(dataset_repo_id)
    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            f"Expected a DatasetDict with train/validation/test splits from "
            f"'{dataset_repo_id}', got {type(dataset)}."
        )
    if split is not None:
        dataset = select_dataset_splits(dataset, split)
    return ensure_validation_split(
        dataset, seed=seed, validation_fraction=validation_fraction
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_type: TaskType,
    sequence_column: str,
    label_column: str,
    max_length: int,
    id_column: str = "id",
    num_proc: int | None = None,
    random_truncate: bool = True,
    random_truncate_by_split: dict[str, bool] | None = None,
    seed: int = 710019,
) -> DatasetDict:
    """Tokenize every split of *dataset* once, ahead of building DataLoaders.

    Uses ``DatasetDict.map(..., batched=True, num_proc=...)`` so tokenization
    happens once (in parallel across ``num_proc`` processes) instead of being
    repeated on every ``EvaluationCollator`` call. For
    ``token_classification``, per-residue labels are aligned here too (``-100``
    at BOS/EOS), so collation only pads.

    Truncation behavior is split-aware:
    - ``random_truncate`` is the fallback applied everywhere when
        ``random_truncate_by_split`` is not provided.
    - ``random_truncate_by_split`` can override truncation mode per split
        (for example random on train, deterministic on validation and test).

    When random truncation is used, each crop window is deterministic per
    example from ``seed`` + dataset index, so results are stable across
    process counts and worker scheduling.

    Leading/trailing special tokens (BOS/EOS or CLS/EOS, whichever the
    tokenizer defines) are re-attached around the cropped residue span via
    tokenizer special token ids instead of ``build_inputs_with_special_tokens``
    (which some custom tokenizers do not implement).
    """
    # Let the caller override the process count; otherwise derive it from the
    # local machine so tokenization stays parallel but bounded.
    num_proc = num_proc if num_proc is not None else default_num_workers() or None

    # Preserve the tokenizer's own special-token convention without relying on
    # build_inputs_with_special_tokens(), which is not implemented uniformly by
    # custom protein tokenizers.
    leading_special_ids = [
        token_id
        for token_id in (tokenizer.bos_token_id, tokenizer.cls_token_id)
        if token_id is not None
    ][:1]
    trailing_special_ids = [
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.sep_token_id)
        if token_id is not None
    ][:1]
    offset = len(leading_special_ids)
    num_special_tokens = offset + len(trailing_special_ids)
    if max_length <= num_special_tokens:
        raise ValueError(
            "max_length must exceed the tokenizer's number of leading and "
            f"trailing special tokens ({num_special_tokens})."
        )
    max_residues = max_length - num_special_tokens
    task = get_task(task_type)

    def _tokenize_batch_manual(
        examples: dict[str, list[Any]],
        indices: list[int],
        random_truncate: bool,
        seed: int,
    ) -> dict[str, list[Any]]:
        # Tokenize without truncation here so we can control the crop window and
        # keep the residue/label alignment deterministic across worker counts.
        residue_ids_batch = tokenizer(
            examples[sequence_column],
            add_special_tokens=False,
            truncation=False,
            # Manual truncation is applied below; suppress tokenizer-level
            # max-length warnings that would otherwise be noisy/misleading.
            verbose=False,
        )["input_ids"]

        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        labels_batch: list[Any] = []
        ids_batch: list[Any] = []
        emit_ids = False

        for i, residue_ids in enumerate(residue_ids_batch):
            num_residues = len(residue_ids)
            start = 0
            if num_residues > max_residues:
                if random_truncate:
                    # Deterministic per-example RNG: same (seed, dataset index)
                    # always yields the same truncation window, regardless of
                    # num_proc or batch composition. A string seed is used
                    # since random.seed() hashes str/bytes via SHA-512
                    # internally, unaffected by PYTHONHASHSEED (unlike a raw
                    # tuple, which random.Random() doesn't accept anyway).
                    rng = random.Random(f"{seed}:{indices[i]}")
                    start = rng.randint(0, num_residues - max_residues)
                residue_ids = residue_ids[start : start + max_residues]

            input_ids = leading_special_ids + residue_ids + trailing_special_ids
            attention_mask = [1] * len(input_ids)
            label = examples[label_column][i] if label_column in examples else None
            if task.aligns_labels:
                label = task.align_labels(
                    label,
                    num_residues=num_residues,
                    start=start,
                    kept=len(residue_ids),
                    offset=offset,
                    input_len=len(input_ids),
                    example_index=indices[i],
                )

            original_id = examples[id_column][i] if id_column in examples else None
            for row in task.expand_tokenized_example(
                input_ids=input_ids,
                attention_mask=attention_mask,
                label=label,
                tokenizer=tokenizer,
                res_start=offset,
                res_end=offset + len(residue_ids),
                example_id=original_id if original_id is not None else indices[i],
            ):
                input_ids_batch.append(row["input_ids"])
                attention_mask_batch.append(row["attention_mask"])
                labels_batch.append(row["label"])
                row_id = row.get("id", original_id)
                if row_id is not None:
                    emit_ids = True
                    ids_batch.append(row_id)
                elif emit_ids:
                    ids_batch.append(None)

        output: dict[str, list[Any]] = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            label_column: labels_batch,
        }
        if emit_ids:
            output[id_column] = ids_batch

        return output

    random_truncate_by_split = random_truncate_by_split or {}
    mapped: dict[str, Any] = {}
    for split_name, split_dataset in dataset.items():
        if (
            task.requires_source_labels
            and label_column not in split_dataset.column_names
        ):
            raise ValueError(
                f"Dataset split '{split_name}' is missing required label "
                f"column '{label_column}' for task_type='{task_type}'."
            )
        split_random_truncate = random_truncate_by_split.get(
            split_name, random_truncate
        )
        mapped[split_name] = split_dataset.map(
            _tokenize_batch_manual,
            batched=True,
            num_proc=num_proc,
            with_indices=True,
            remove_columns=split_dataset.column_names,
            fn_kwargs={
                "random_truncate": split_random_truncate,
                "seed": seed if split_random_truncate else 0,
            },
        )
    return DatasetDict(mapped)


def build_dataloaders(
    dataset: DatasetDict,
    collator: EvaluationCollator,
    config: DataLoaderConfig,
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    """Build train/validation/test :class:`DataLoader` instances from a loaded dataset.

    Any split absent from *dataset* yields ``None`` for that DataLoader.
    """
    # Keep DataLoader workers aligned with the tokenizer worker policy unless
    # the caller explicitly pins a value in the config.
    num_workers = (
        config.num_workers if config.num_workers is not None else default_num_workers()
    )

    common_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if num_workers > 0 else False,
    )

    def _build(
        split: str, shuffle: bool, generator: torch.Generator | None = None
    ) -> DataLoader | None:
        if split not in dataset:
            return None
        if config.max_tokens_per_batch is not None:
            assert generator is not None
            lengths = [len(ids) for ids in dataset[split]["input_ids"]]
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
            # Reshuffles mega-batches every __iter__ regardless of `shuffle`,
            # so eval order needs a seeded generator too for reproducibility.
            lengths = [len(ids) for ids in dataset[split]["input_ids"]]
            sampler = LengthGroupedSampler(
                lengths, config.batch_size, generator=generator
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
