"""Feature-cache settings, identity, storage and integrity for frozen-trunk features.

This module owns *where and how* extracted features are stored, keyed and
validated on disk. It deliberately does not import predictors or tasks:
callers pass explicit extraction settings and tensors in. Feature extraction
itself lives in :mod:`modules.evaluate.src.predictors.embeddings`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

EmbeddingPooling = Literal["mean", "cls"]
EmbeddingDtype = Literal["float16", "float32"]
EmbeddingAutocastDtype = Literal["float16", "bfloat16", "float32"]

_ITEMSIZE = {"float16": 2, "float32": 4}
BYTES_PER_GB = 1024**3

# Splits eligible for the on-disk cache. Test embeddings are shared by probe
# variants and model-based controls, so they are worth persisting too.
CACHEABLE_SPLITS = frozenset({"train", "validation", "test"})


class EmbeddingCacheConfig(BaseModel):
    """Explicit feature-cache settings consumed by the prepare step."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    pooling: EmbeddingPooling = Field(
        "mean",
        description=(
            "How to pool (B, S, H) trunk hidden states into (B, H). Must "
            "use 'mean' for residue-only masked mean pooling, or 'cls' "
            "for first-token pooling."
        ),
    )
    layers: list[int] | None = Field(
        default=None,
        description=(
            "Indices into the model's returned hidden-state tuple. Negative "
            "indices follow Python indexing; multiple layers are pooled "
            "independently then concatenated. If omitted while the cache is "
            "enabled, [-1] selects the final layer."
        ),
    )
    dtype: EmbeddingDtype = Field(
        "float32", description="Precision embeddings are stored/computed in."
    )
    autocast_dtype: EmbeddingAutocastDtype = Field(
        "float32",
        description=(
            "Precision used inside the frozen-trunk forward pass. float32 "
            "disables autocast; float16/bfloat16 autocast is opt-in."
        ),
    )
    normalize_before_pooling: bool = Field(
        True,
        description=(
            "L2-normalize each token representation before mean pooling. "
            "Disable explicitly to retain raw representation magnitudes."
        ),
    )
    max_cache_gb: float = Field(
        2.0,
        description=(
            "Refuse to write a split's embeddings to disk if they would "
            "exceed this many GB; falls back to an in-memory-only cache for "
            "that run instead of risking filling shared storage."
        ),
    )
    cache_policy: Literal["reuse", "refresh", "off"] = Field(
        "reuse",
        description=(
            "'reuse': load a matching cached split from disk if present. "
            "'refresh': always recompute and overwrite. 'off': compute "
            "in-memory only, never read or write the disk cache."
        ),
    )

    @field_validator("layers")
    @classmethod
    def _validate_layers(cls, layers: list[int] | None) -> list[int] | None:
        if layers is None:
            return None
        if not layers:
            raise ValueError(
                "embedding_cache.layers must contain at least one layer index."
            )
        if len(set(layers)) != len(layers):
            raise ValueError(
                "embedding_cache.layers must not contain duplicate indices."
            )
        return layers

    @property
    def resolved_layers(self) -> list[int]:
        """Return the effective layers for an enabled embedding cache."""
        return self.layers if self.layers is not None else [-1]


def embedding_cache_key(
    *,
    model_id: str,
    dataset_id: str,
    split: str,
    max_length: int,
    random_truncate: bool,
    seed: int,
    pooling: EmbeddingPooling,
    dtype: EmbeddingDtype,
    layers: Sequence[int] = (-1,),
    model_fingerprint: str | None = None,
    dataset_fingerprint: str | None = None,
    autocast_dtype: EmbeddingAutocastDtype | None = None,
    normalize_before_pooling: bool = False,
    input_transform: str | None = None,
    packed: bool = False,
    length_grouped_sampling: bool = False,
) -> str:
    """Content hash identifying a cached embedding tensor.

    Every setting that changes the embedding's content is folded into the key,
    so changing pooling, max_length, or float32 backend precision cannot
    silently reuse stale features.

    ``model_id``/``dataset_id`` alone only identify a Hub *repository*, not
    its content: since Hub revisions default to a mutable "main" branch, a
    later push to the same model/dataset repo (new weights, new/edited
    examples) would otherwise silently reuse embeddings computed from the
    old content. ``model_fingerprint``/``dataset_fingerprint`` (a weights
    hash and a dataset content hash, respectively) close that gap; both
    default to ``None`` when generating an isolated key without content
    fingerprints.
    """
    payload = json.dumps(
        {
            "cache_format": 5,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "split": split,
            "max_length": max_length,
            "random_truncate": random_truncate,
            # The seed only changes the embedding content when random
            # truncation is enabled. Deterministic splits should reuse the
            # same cache entry across pipeline seeds.
            "seed": seed if random_truncate else None,
            "pooling": pooling,
            "dtype": dtype,
            "layers": list(layers),
            "model_fingerprint": model_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "autocast_dtype": autocast_dtype,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "normalize_before_pooling": normalize_before_pooling,
            "input_transform": input_transform,
            "packed": packed,
            "length_grouped_sampling": length_grouped_sampling,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def weights_fingerprint(model: nn.Module) -> str:
    """Content digest of *model*'s parameters for cache invalidation."""
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
        hasher.update(tensor.numpy().tobytes())
    return hasher.hexdigest()[:16]


def estimate_embedding_bytes(
    num_examples: int, hidden_size: int, dtype: EmbeddingDtype
) -> int:
    return num_examples * hidden_size * _ITEMSIZE[dtype]


class EmbeddingCache:
    """Reads/writes one split's pooled embeddings as a single safetensors file."""

    def __init__(self, cache_dir: Path, max_cache_gb: float) -> None:
        self.cache_dir = cache_dir
        self.max_cache_bytes = int(max_cache_gb * BYTES_PER_GB)

    def _path(self, split: str, key: str) -> Path:
        return self.cache_dir / f"{split}_{key}.safetensors"

    def exists(self, split: str, key: str) -> bool:
        return self._path(split, key).exists()

    def fits_budget(
        self, num_examples: int, hidden_size: int, dtype: EmbeddingDtype
    ) -> bool:
        return (
            estimate_embedding_bytes(num_examples, hidden_size, dtype)
            <= self.max_cache_bytes
        )

    def write(
        self,
        split: str,
        key: str,
        embeddings: torch.Tensor,
        ids: list[Any],
        labels: torch.Tensor,
    ) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(split, key)
        # Write to a temp path first so a crash mid-write can never leave a
        # cache file that `exists()` treats as valid but is truncated/corrupt.
        tmp_path = path.with_suffix(".tmp")
        save_file(
            {"embeddings": embeddings.contiguous(), "labels": labels.contiguous()},
            str(tmp_path),
            metadata={"ids": json.dumps(ids)},
        )
        tmp_path.replace(path)
        return path

    def read(
        self, split: str, key: str
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor]:
        # safetensors memory-maps the file rather than loading it eagerly, so
        # reading a cached split doesn't spike host RAM the way a plain
        # torch.load of a large tensor would.
        with safe_open(str(self._path(split, key)), framework="pt", device="cpu") as f:
            embeddings = f.get_tensor("embeddings")
            labels = f.get_tensor("labels")
            metadata = f.metadata() or {}
        if "ids" not in metadata:
            raise ValueError(
                f"Embedding cache entry {self._path(split, key)} has no IDs."
            )
        ids = json.loads(metadata["ids"])
        return embeddings, ids, labels
