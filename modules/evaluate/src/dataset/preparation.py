"""Sequence data preparation: config schema, tokenization and token DataLoaders.

This module owns how a raw evaluation ``DatasetDict`` becomes tokenized,
collated DataLoaders for every role (train, validation, test, and configured
final-refit splits). Model loading and frozen-trunk embedding caching are owned
by the predictors and are not implemented here.
"""

from __future__ import annotations

from typing import Any

from accelerate import Accelerator
from datasets import DatasetDict
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.artifacts.feature_cache import EmbeddingCacheConfig
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    DatasetSplitConfig,
    build_dataloaders,
    tokenize_dataset,
)
from modules.evaluate.src.dataset.workspace import TaskType


class PrepareConfig(BaseModel):
    """Config. for the prepare step in evaluation."""

    model_config = ConfigDict(extra="forbid")
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    batch_size: int = 16
    packed: bool = False
    max_tokens_per_batch: int | None = Field(None, gt=0)
    embedding_batch_size: int = Field(
        128,
        gt=0,
        description=(
            "Batch size used only while extracting frozen-trunk embeddings. "
            "Increase until the trunk approaches device memory capacity."
        ),
    )
    dataloader_num_workers: int = Field(
        8,
        ge=0,
        description=(
            "Number of dataloader worker processes per rank and per loader. "
            "Defaults to 8; set to 0 to load batches in the main process."
        ),
    )
    max_length: int = Field(
        512, description="Maximum sequence length after truncation."
    )
    pad_to_multiple_of: int | None = 8
    sequence_column: str = "sequence"
    label_column: str = Field(
        "targets",
        description=(
            "Dataset column containing label(s). Standardized evaluation "
            "datasets use 'targets', not the source field 'label'."
        ),
    )
    id_column: str = Field(
        "id",
        description=(
            "Dataset column containing a per-example identifier, passed "
            "through the batch so downstream prediction/scoring steps can key "
            "predictions/labels back to their source example. Falls back "
            "to a positional index when the dataset has no such column."
        ),
    )
    tokenize_num_proc: int = Field(
        8,
        gt=0,
        description=(
            "Number of processes used to tokenize the dataset up front. "
            "Defaults to 8; must be positive and is independent of dataloader workers."
        ),
    )
    random_truncate_by_split: dict[str, bool] = Field(
        default_factory=lambda: {
            "train": True,
            "validation": False,
            "test": False,
        },
        description=(
            "Whether sequences longer than max_length are cropped from a "
            "random window, keyed by split. Defaults to random train crops "
            "and deterministic validation/test crops."
        ),
    )
    validation_fraction: float = Field(
        0.1,
        description=(
            "Fraction of 'train' held out to synthesize a 'validation' split "
            "when the dataset doesn't already have one."
        ),
    )
    split: DatasetSplitConfig = Field(
        default_factory=DatasetSplitConfig,
        description=(
            "Named dataset split method and source split names to map onto "
            "the pipeline's train/validation/test roles and final-refit-only data."
        ),
    )
    length_grouped_sampling: bool = Field(
        False,
        description=(
            "Batch examples of similar tokenized length together (via "
            "LengthGroupedSampler) instead of pure random shuffling. Reduces "
            "the amount of padding the collator has to add per batch, at the "
            "cost of batches no longer being uniformly randomly composed. "
            "Frozen-trunk extraction groups lengths deterministically; "
            "cached feature batches are not grouped."
        ),
    )
    embedding_cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)


def build_collator(
    config: PrepareConfig, tokenizer: PreTrainedTokenizerBase, task_type: TaskType
) -> EvaluationCollator:
    """Collator producing token batches for *task_type*."""
    return EvaluationCollator(
        tokenizer=tokenizer,
        task_type=task_type,
        config=EvaluationCollatorConfig(
            pad_to_multiple_of=config.pad_to_multiple_of,
            packed=config.packed,
            label_column=config.label_column,
            id_column=config.id_column,
        ),
    )


def dataloader_config(config: PrepareConfig, seed: int) -> DataLoaderConfig:
    """DataLoader settings shared by every token split of one run."""
    return DataLoaderConfig(
        batch_size=config.batch_size,
        max_tokens_per_batch=config.max_tokens_per_batch,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.dataloader_persistent_workers,
        seed=seed,
        length_grouped_sampling=config.length_grouped_sampling,
    )


def tokenize_split_dataset(
    config: PrepareConfig,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_type: TaskType,
    seed: int,
    accelerator: Accelerator,
) -> DatasetDict:
    """Tokenize every split once. Rank 0 fills the shared cache first."""
    with accelerator.main_process_first():
        return tokenize_dataset(
            dataset,
            tokenizer=tokenizer,
            task_type=task_type,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            max_length=config.max_length,
            id_column=config.id_column,
            num_proc=config.tokenize_num_proc,
            random_truncate_by_split=config.random_truncate_by_split,
            seed=seed,
        )


def build_token_dataloaders(
    config: PrepareConfig,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_type: TaskType,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Train, validation and test token DataLoaders (``None`` when absent)."""
    return build_dataloaders(
        dataset=dataset,
        collator=build_collator(config, tokenizer, task_type),
        config=dataloader_config(config, seed),
    )


def build_refit_token_dataloaders(
    config: PrepareConfig,
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    task_type: TaskType,
    seed: int,
    accelerator: Accelerator,
) -> dict[str, Any]:
    """Accelerator-prepared token loaders for each configured final-refit split."""
    collator = build_collator(config, tokenizer, task_type)
    refit_token_dataloaders: dict[str, Any] = {}
    for split in config.split.refit_additional_splits:
        if split not in dataset:
            raise ValueError(
                f"Configured final-refit source split '{split}' was not loaded. "
                f"Available splits: {sorted(dataset.keys())}."
            )
        split_dataset = DatasetDict({"train": dataset[split]})
        refit_loader, _, _ = build_dataloaders(
            dataset=split_dataset,
            collator=collator,
            config=dataloader_config(config, seed),
        )
        if refit_loader is None:
            raise ValueError(f"Configured final-refit source split '{split}' is empty.")
        refit_token_dataloaders[split] = accelerator.prepare(refit_loader)
    return refit_token_dataloaders
