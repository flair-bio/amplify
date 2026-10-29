"""Materialized feature datasets, retained extraction context, and loaders."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.evaluate.src.dataset.dataloader import LengthGroupedSampler, token_lengths
from modules.evaluate.src.dataset.embedding_cache import (
    BYTES_PER_GB,
    CACHEABLE_SPLITS,
    EmbeddingCache,
    embedding_cache_key,
    estimate_embedding_bytes,
    weights_fingerprint,
)
from modules.evaluate.src.model.embeddings import AUTOCAST_DTYPES, extract_embeddings

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import (
        EmbeddingAutocastDtype,
        EmbeddingCacheConfig,
        EmbeddingDtype,
        EmbeddingPooling,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBackend:
    """Retained trunk and token loaders for perturbed-input/model controls."""

    trunk: nn.Module
    token_test_dataloader: DataLoader
    pooling: EmbeddingPooling
    layers: list[int]
    dtype: EmbeddingDtype
    hidden_size: int
    autocast_dtype: EmbeddingAutocastDtype = "float32"
    normalize_before_pooling: bool = False
    test_cache_key: str | None = None
    cache: EmbeddingCache | None = None
    cache_policy: Literal["reuse", "refresh", "off"] = "reuse"
    model_id: str | None = None
    dataset_id: str | None = None
    max_length: int | None = None
    test_random_truncate: bool = False
    test_dataset_fingerprint: str | None = None
    length_grouped_sampling: bool = False
    token_train_dataloader: DataLoader | None = None
    token_val_dataloader: DataLoader | None = None
    train_random_truncate: bool = True
    validation_random_truncate: bool = False
    extraction_batch_size: int | None = None

    def test_embedding_key(
        self,
        *,
        trunk: nn.Module | None = None,
        seed: int | None = None,
        input_transform: str | None = None,
    ) -> str | None:
        """Return the clean or transformed test embedding key."""
        if trunk is None and input_transform is None:
            return self.test_cache_key
        if (
            self.model_id is None
            or self.dataset_id is None
            or self.max_length is None
            or trunk is None
            or seed is None
        ):
            return None
        return embedding_cache_key(
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            split="test",
            max_length=self.max_length,
            random_truncate=self.test_random_truncate,
            seed=seed,
            pooling=self.pooling,
            dtype=self.dtype,
            layers=self.layers or [-1],
            model_fingerprint=weights_fingerprint(getattr(trunk, "base_model", trunk)),
            dataset_fingerprint=self.test_dataset_fingerprint,
            autocast_dtype=self.autocast_dtype,
            normalize_before_pooling=self.normalize_before_pooling,
            input_transform=input_transform,
            packed=getattr(self.token_test_dataloader.collate_fn, "packed", False),
            length_grouped_sampling=self.length_grouped_sampling,
        )


class EmbeddingDataset(Dataset):
    """A cached ``(N, H)`` tensor with aligned IDs and labels."""

    def __init__(
        self, embeddings: torch.Tensor, ids: list[Any], labels: list[Any] | torch.Tensor
    ) -> None:
        assert len(embeddings) == len(ids) == len(labels)
        self.embeddings = embeddings
        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "inputs_embeds": self.embeddings[index],
            "labels": self.labels[index],
            "id": self.ids[index],
        }


def find_embedding_dataset(dataloader: Any) -> EmbeddingDataset | None:
    """Find a materialized embedding dataset behind loader wrappers."""
    current = dataloader
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, EmbeddingDataset):
            return current
        seen.add(id(current))
        current = getattr(current, "dataset", None)
    return None


def embedding_collate_fn(
    batch: list[dict[str, Any]], label_dtype: torch.dtype
) -> dict[str, Any]:
    """Stack cached embeddings and labels in the task's label dtype."""
    return {
        "inputs_embeds": torch.stack([example["inputs_embeds"] for example in batch]),
        "labels": torch.stack([example["labels"] for example in batch]).to(label_dtype),
        "id": [example["id"] for example in batch],
    }


def build_embedding_dataloaders(
    *,
    model: nn.Module,
    splits: dict[str, tuple[DataLoader | None, bool]],
    accelerator: Accelerator,
    cache: EmbeddingCache,
    cache_config: EmbeddingCacheConfig,
    model_id: str,
    dataset_id: str,
    max_length: int,
    seed: int,
    batch_size: int,
    embedding_size: int,
    label_dtype: torch.dtype,
    extraction_batch_size: int | None = None,
    is_ragged: bool = False,
    model_fingerprint: str | None = None,
    length_grouped_sampling: bool = False,
) -> dict[str, DataLoader | None]:
    """Load or extract each split and build accelerator-prepared feature loaders.

    Extraction may use deterministic length grouping; feature batches do not.
    Ragged embeddings stay in memory regardless of disk-cache policy.
    """
    embedding_dataloaders: dict[str, DataLoader | None] = {}
    cache_layers: Sequence[int] = (
        cache_config.layers if cache_config.layers is not None else (-1,)
    )
    model_fingerprint = model_fingerprint or weights_fingerprint(model)
    for split, (dataloader, random_truncate) in splits.items():
        if dataloader is None:
            embedding_dataloaders[split] = None
            continue

        dataset_fingerprint = getattr(dataloader.dataset, "_fingerprint", None)
        key = embedding_cache_key(
            model_id=model_id,
            dataset_id=dataset_id,
            split=split,
            max_length=max_length,
            random_truncate=random_truncate,
            seed=seed,
            pooling=cache_config.pooling,
            dtype=cache_config.dtype,
            layers=cache_layers,
            model_fingerprint=model_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            autocast_dtype=cache_config.autocast_dtype,
            normalize_before_pooling=cache_config.normalize_before_pooling,
            packed=getattr(dataloader.collate_fn, "packed", False),
            length_grouped_sampling=length_grouped_sampling,
        )
        cacheable = (
            not is_ragged
            and split in CACHEABLE_SPLITS
            and cache_config.cache_policy != "off"
        )

        embeddings: torch.Tensor
        ids: list[Any]
        labels: torch.Tensor
        if (
            cacheable
            and cache_config.cache_policy == "reuse"
            and cache.exists(split, key)
        ):
            if accelerator.is_main_process:
                logger.info("Loading cached '%s' embeddings from disk...", split)
            embeddings, ids, labels = cache.read(split, key)
        else:
            if accelerator.is_main_process:
                logger.info(
                    "Extracting '%s' embeddings from the frozen trunk...", split
                )
                if is_ragged and "input_ids" in getattr(
                    dataloader.dataset, "column_names", ()
                ):
                    num_tokens = sum(token_lengths(dataloader.dataset))
                    logger.info(
                        "Token-level '%s' embeddings will hold up to %.1f GB in "
                        "memory (%d tokens x %d features).",
                        split,
                        estimate_embedding_bytes(
                            num_tokens, embedding_size, cache_config.dtype
                        )
                        / BYTES_PER_GB,
                        num_tokens,
                        embedding_size,
                    )
            extraction_dataloader = dataloader
            if extraction_batch_size is not None:
                sampler = None
                if length_grouped_sampling:
                    lengths = token_lengths(dataloader.dataset)
                    sampler = LengthGroupedSampler(
                        lengths, extraction_batch_size, shuffle=False
                    )
                extraction_dataloader = accelerator.prepare(
                    DataLoader(
                        dataloader.dataset,
                        batch_size=extraction_batch_size,
                        sampler=sampler,
                        shuffle=False,
                        collate_fn=dataloader.collate_fn,
                        num_workers=dataloader.num_workers,
                        pin_memory=dataloader.pin_memory,
                        persistent_workers=(
                            dataloader.persistent_workers
                            if dataloader.num_workers > 0
                            else False
                        ),
                    )
                )
            embeddings, ids, labels, _ = extract_embeddings(
                model,
                extraction_dataloader,
                accelerator,
                cache_config.pooling,
                cache_config.dtype,
                layers=cache_layers,
                is_ragged=is_ragged,
                autocast_dtype=AUTOCAST_DTYPES[cache_config.autocast_dtype],
                normalize_before_pooling=cache_config.normalize_before_pooling,
                progress_label=f"Extracting '{split}' embeddings",
            )
            if cacheable and accelerator.is_main_process:
                if cache.fits_budget(
                    embeddings.shape[0], embedding_size, cache_config.dtype
                ):
                    cache.write(split, key, embeddings, ids, labels)
                else:
                    logger.warning(
                        "'%s' embeddings (%d examples x %d) exceed "
                        "max_cache_gb=%.2f; using them for this run only, "
                        "not writing to disk.",
                        split,
                        embeddings.shape[0],
                        embedding_size,
                        cache_config.max_cache_gb,
                    )

        embedding_dataset = EmbeddingDataset(embeddings, ids, labels.to(label_dtype))
        embedding_dataloaders[split] = accelerator.prepare(
            DataLoader(
                embedding_dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                generator=(
                    torch.Generator().manual_seed(seed) if split == "train" else None
                ),
                collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
            )
        )
    return embedding_dataloaders
