"""Representation cache identities, storage, validation, and size policy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from modules.evaluate.src.utils.storage import atomic_output, cache_lock

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import (
        EmbeddingAutocastDtype,
        EmbeddingDtype,
        EmbeddingPooling,
    )

BYTES_PER_GB = 1024**3
_ITEMSIZE = {"float16": 2, "float32": 4}
CACHEABLE_SPLITS = frozenset({"train", "validation", "test"})


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
    """Hash every setting that changes a split's embedding content."""
    payload = json.dumps(
        {
            "cache_format": 5,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "split": split,
            "max_length": max_length,
            "random_truncate": random_truncate,
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
    """Content digest of model parameters for cache invalidation."""
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


def tensor_cache_key(identity: Mapping[str, Any]) -> str:
    """Hash an explicit, JSON-serializable representation identity."""
    return hashlib.sha256(
        json.dumps(dict(identity), sort_keys=True).encode()
    ).hexdigest()


def validate_tensor_cache(
    values: Mapping[str, torch.Tensor],
    *,
    shapes: Mapping[str, tuple[int, ...]],
    dtype: torch.dtype,
) -> None:
    """Require exactly the expected tensors, shapes, dtype, and finite values."""
    if set(values) != set(shapes):
        raise ValueError("Tensor cache has inconsistent tensor identities")
    for name, shape in shapes.items():
        tensor = values[name]
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(
                f"Tensor cache has invalid shape, dtype, or values for {name!r}"
            )


def load_tensor_cache(
    *,
    cache_dir: Path | None,
    namespace: str,
    identity: Mapping[str, Any],
    extract: Callable[[], dict[str, torch.Tensor]],
    validate: Callable[[dict[str, torch.Tensor]], None],
    key: str | None = None,
    refresh: bool = False,
    max_cache_gb: float = 50,
) -> tuple[dict[str, torch.Tensor], str]:
    """Load or extract tensors, validating both paths before returning or writing.

    The caller owns task-specific identity and validation. ``key`` may name an
    existing cache entry; otherwise it is derived from the supplied identity.
    """
    key = tensor_cache_key(identity) if key is None else key
    if cache_dir is None:
        values = extract()
        validate(values)
        return values, key
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{namespace}-{key}.safetensors"
    identity_json = json.dumps(dict(identity), sort_keys=True)
    with cache_lock(path.with_suffix(".lock")):
        if path.exists() and not refresh:
            with safe_open(str(path), framework="pt", device="cpu") as cached:
                stored_identity = (cached.metadata() or {}).get("identity")
                if stored_identity is None or json.loads(stored_identity) != json.loads(
                    identity_json
                ):
                    raise ValueError(f"Tensor cache {path} has a mismatched identity")
                values = {name: cached.get_tensor(name) for name in cached.keys()}
            validate(values)
        else:
            values = extract()
            validate(values)
            size = sum(
                tensor.numel() * tensor.element_size() for tensor in values.values()
            )
            if size <= max_cache_gb * BYTES_PER_GB:
                with atomic_output(path) as temporary:
                    save_file(
                        values,
                        str(temporary),
                        metadata={"identity": identity_json},
                    )
    return values, key


class EmbeddingCache:
    """Read/write a split's pooled embeddings as a single safetensors file."""

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
        with atomic_output(path) as temporary:
            save_file(
                {"embeddings": embeddings.contiguous(), "labels": labels.contiguous()},
                str(temporary),
                metadata={"ids": json.dumps(ids)},
            )
        return path

    def read(
        self, split: str, key: str
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor]:
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
