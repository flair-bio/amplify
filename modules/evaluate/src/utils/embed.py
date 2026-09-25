"""
Cache frozen-trunk embeddings so a linear-probe run only pays for one
trunk forward pass per split instead of one per HPO trial/epoch. Requires
a frozen trunk (``mode='tune_probe'``): with the trunk frozen, its output
for a given tokenized example is identical across every epoch and every
hyperparameter trial, so it only needs to be computed once.

Sequence-level tasks (``sequence_classification``, ``sequence_regression``)
pool each example down to one ``(hidden_size,)`` vector. Ragged/token-level
tasks (``token_classification``) skip pooling and keep one vector per valid
token instead -- see ``TaskHandler.is_ragged``. Token-level embeddings are
~100-1000x larger than pooled sequence embeddings (total_tokens x
hidden_size vs. num_examples x hidden_size), so unlike sequence-level
splits they are never written to the on-disk cache, only held in memory for
the current run.

Storage stays small on purpose: only the pooled ``(num_examples,
hidden_size)`` tensor is cached (a few hundred MB at most for realistic
dataset sizes), written once per split as a single safetensors file (not
one file per example, which would blow up inode counts on shared/cluster
filesystems), in float16 by default. A byte-budget check runs before every
write so a misconfigured run degrades to "recompute, don't cache" instead
of filling the disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import torch
from accelerate import Accelerator
from pydantic import BaseModel, ConfigDict, Field, field_validator
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modules.evaluate.src.dataset.collator import input_id_rows, unpack_packed
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.utils.inference import pad_for_gather

logger = logging.getLogger(__name__)

EmbeddingPooling = Literal["mean", "cls"]
EmbeddingDtype = Literal["float16", "float32"]
EmbeddingAutocastDtype = Literal["float16", "bfloat16", "float32"]

_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}
_AUTOCAST_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_ITEMSIZE = {"float16": 2, "float32": 4}

# Splits eligible for the on-disk cache. Test embeddings are shared by probe
# variants and model-based controls, so they are worth persisting too.
CACHEABLE_SPLITS = frozenset({"train", "validation", "test"})


class EmbeddingCacheConfig(BaseModel):
    """Config for the frozen-trunk embedding cache used by :class:`PrepareStep`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    pooling: EmbeddingPooling = Field(
        "mean",
        description=(
            "How to pool (B, S, H) trunk hidden states into (B, H). Must "
            "match the downstream head's own pooling to give equivalent "
            "results: 'mean' for AMPLIFY-style masked mean pooling, 'cls' "
            "for ESM-style CLS-token pooling."
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
        "float16", description="Precision embeddings are stored/computed in."
    )
    autocast_dtype: EmbeddingAutocastDtype = Field(
        "float16",
        description=(
            "Precision used inside the frozen-trunk forward pass. float32 "
            "disables autocast; float16 is the fast GPU default."
        ),
    )
    normalize_before_pooling: bool = Field(
        False,
        description=(
            "L2-normalize each token representation before mean pooling. "
            "Enable for compatibility with v0's normalized embeddings."
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


def validate_embedding_cache_compat(
    is_ragged: bool, tokenizer: Any, pooling: EmbeddingPooling
) -> None:
    """Raise if *tokenizer*/*pooling* are incompatible with the embedding
    cache, before any (expensive) trunk forward pass runs.

    ``pooling`` only applies to non-ragged (sequence-level) tasks: ragged
    tasks (e.g. ``token_classification``) keep one embedding per valid
    token instead of pooling, so the 'cls' position-0 assumption below
    doesn't apply to them.
    """
    if is_ragged:
        return
    if pooling == "cls" and (
        tokenizer.cls_token_id is None and tokenizer.bos_token_id is None
    ):
        raise ValueError(
            "embedding_cache.pooling='cls' pools position 0 of the "
            f"trunk's hidden states, but tokenizer '{tokenizer.__class__.__name__}' "
            "defines neither a cls_token nor a bos_token, so position 0 "
            "is not guaranteed to be a fixed special token. Use "
            "pooling='mean' for this tokenizer instead."
        )


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: EmbeddingPooling,
    normalize_before_pooling: bool = False,
) -> torch.Tensor:
    """Pool ``(B, S, H)`` trunk hidden states to ``(B, H)``.

    'mean' mirrors ``AMPLIFYForSequenceClassification.forward``'s masked
    mean pooling exactly, so cached embeddings are numerically equivalent
    to what the un-cached model would have pooled internally.
    """
    if normalize_before_pooling:
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
    if pooling == "cls":
        return hidden_states[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _select_layers(
    hidden_states: Sequence[torch.Tensor], layers: Sequence[int]
) -> list[torch.Tensor]:
    try:
        return [hidden_states[layer] for layer in layers]
    except IndexError as exc:
        available = len(hidden_states)
        raise ValueError(
            f"Requested embedding layer(s) {list(layers)} but the model returned "
            f"{available} hidden states (valid indices are {-available} through "
            f"{available - 1})."
        ) from exc


def pool_hidden_state_layers(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    pooling: EmbeddingPooling,
    layers: Sequence[int],
    normalize_before_pooling: bool = False,
) -> torch.Tensor:
    """Pool selected hidden-state layers and concatenate them feature-wise."""
    selected = _select_layers(hidden_states, layers)
    return torch.cat(
        [
            pool_hidden_states(
                layer,
                attention_mask,
                pooling,
                normalize_before_pooling=normalize_before_pooling,
            )
            for layer in selected
        ],
        dim=-1,
    )


def select_token_hidden_state_layers(
    hidden_states: Sequence[torch.Tensor], layers: Sequence[int]
) -> torch.Tensor:
    """Concatenate selected hidden-state layers feature-wise, without
    pooling over the sequence dimension: ``(B, S, H)`` per layer becomes
    ``(B, S, H * len(layers))``. Used for ragged/token-level tasks, where
    each token (not each example) needs its own embedding."""
    selected = _select_layers(hidden_states, layers)
    return torch.cat(selected, dim=-1)


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
) -> str:
    """Content hash identifying a cached embedding tensor.

    Every setting that changes the embedding's content is folded into the key,
    so changing pooling or max_length cannot silently reuse stale features.

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
            "cache_format": 3,
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
            "normalize_before_pooling": normalize_before_pooling,
            "input_transform": input_transform,
            "packed": packed,
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


def _autocast_context(
    accelerator: Accelerator, autocast_dtype: torch.dtype | None
) -> Any:
    """Return a safe autocast context for the accelerator's device."""
    device = getattr(accelerator, "device", None)
    device_type = getattr(device, "type", None)
    if (
        autocast_dtype is None
        or autocast_dtype == torch.float32
        or device_type is None
        or device_type == "cpu"
        and autocast_dtype != torch.bfloat16
    ):
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=autocast_dtype)


class EmbeddingCache:
    """Reads/writes one split's pooled embeddings as a single safetensors file."""

    def __init__(self, cache_dir: Path, max_cache_gb: float) -> None:
        self.cache_dir = cache_dir
        self.max_cache_bytes = int(max_cache_gb * (1024**3))

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


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    accelerator: Accelerator,
    pooling: EmbeddingPooling,
    dtype: EmbeddingDtype,
    layers: Sequence[int] = (-1,),
    batch_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    is_ragged: bool = False,
    label_ignore_index: int = -100,
    autocast_dtype: torch.dtype | None = None,
    normalize_before_pooling: bool = False,
    gather_across_processes: bool = True,
) -> tuple[torch.Tensor, list[Any], torch.Tensor, list[str] | None]:
    """One frozen forward pass over *dataloader*, returning embeddings
    (float `dtype`, on CPU), their aligned ids, labels, and (if *tokenizer*
    is given) each example's decoded input sequence.

    For non-ragged (sequence-level) tasks, returns one pooled ``(N, H)`` row
    per example. For ragged/token-level tasks (``is_ragged=True``), pooling
    is skipped and one row is returned per valid token instead (positions
    where ``labels != label_ignore_index``, i.e. excluding padding/special
    tokens) -- ``ids`` are duplicated per valid token of their example so
    every row still traces back to its source example. Re-embedding
    controls (``tokenizer``/``batch_transform``, used by the
    ``scrambled_sequences`` control) assume one row per example and are not
    supported when ``is_ragged=True``.

    Labels are captured from the same pass/order as the embeddings (rather
    than by iterating the dataloader again separately) since a shuffled
    dataloader (e.g. the train split) yields a different example order on
    each fresh iteration, which would otherwise desynchronize labels from
    the embeddings/ids captured here.

    ``batch_transform``, when set, is applied to each (token) batch after
    popping ``id``/``labels`` but before the forward pass -- used by
    PredictStep's ``scrambled_sequences`` control to re-embed corrupted
    inputs through the same (unchanged) trunk.

    Runs the wrapper's ``base_model`` (the pretrained trunk), deliberately
    bypassing the task-specific classifier. When only the final layer is
    requested, it uses ``last_hidden_state`` without requesting the complete
    hidden-state tuple. Multiple or non-final layers use
    ``output_hidden_states=True``. ``autocast_dtype`` applies to the trunk
    forward, not just to the stored CPU embeddings.
    """
    if is_ragged and tokenizer is not None:
        raise ValueError(
            "extract_embeddings(is_ragged=True) flattens rows to one per "
            "valid token, so decoding a per-example sequence (tokenizer=...) "
            "is not supported."
        )
    model.eval()
    torch_dtype = _TORCH_DTYPE[dtype]
    all_ids: list[Any] = []
    all_labels: list[Any] = []
    all_sequences: list[str] | None = [] if tokenizer is not None else None
    has_ids = False
    trunk = getattr(model, "base_model", model)
    trunk.eval()

    # `accelerator.gather_for_metrics` returns the full (deduplicated) dataset
    # length across every batch iterated, so when the dataset length is known
    # up front we can write pooled embeddings directly into a preallocated
    # (N, H) buffer instead of accumulating a per-batch list and doubling peak
    # host RAM at the final `torch.cat`. This matters most on large datasets
    # (>100k sequences), where that transient doubling risks an OOM crash.
    # Not applicable when is_ragged=True: the row count (valid tokens) isn't
    # known until each batch's labels are inspected, so that path always
    # accumulates chunks instead.
    try:
        num_examples: int | None = (
            None if is_ragged else len(dataloader.dataset)  # type: ignore[arg-type]
        )
    except (AttributeError, TypeError):
        num_examples = None
    embeddings: torch.Tensor | None = None
    embeddings_chunks: list[torch.Tensor] = []
    offset = 0
    for batch in dataloader:
        batch = dict(batch)
        ids = batch.pop("id", None)
        labels = batch.pop("labels")
        if batch_transform is not None:
            batch = batch_transform(batch)
        if not gather_across_processes:
            batch = {
                key: (
                    value.to(device=accelerator.device)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in batch.items()
            }
            labels = labels.to(device=accelerator.device)
        use_final_layer_fast_path = len(layers) == 1 and layers[0] == -1
        with _autocast_context(accelerator, autocast_dtype):
            outputs = trunk(**batch, output_hidden_states=not use_final_layer_fast_path)
        if use_final_layer_fast_path:
            final_hidden_state = getattr(outputs, "last_hidden_state", None)
            if final_hidden_state is None:
                raise ValueError(
                    "The trunk must return last_hidden_state when extracting "
                    "only the final embedding layer."
                )
            hidden_states: Sequence[torch.Tensor] = (final_hidden_state,)
        else:
            hidden_states = outputs.hidden_states
        if "cu_seqlens" in batch:
            cu_seqlens = batch["cu_seqlens"]
            num_sequences = batch["num_sequences"]
            hidden_states = tuple(
                unpack_packed(layer, cu_seqlens, num_sequences)
                for layer in hidden_states
            )
            lengths = (
                cu_seqlens[1 : num_sequences + 1] - cu_seqlens[:num_sequences]
            ).long()
            attention_mask = (
                torch.arange(hidden_states[0].shape[1], device=lengths.device)[None, :]
                < lengths[:, None]
            )
            if is_ragged:
                labels = unpack_packed(
                    labels, cu_seqlens, num_sequences, label_ignore_index
                )
        else:
            attention_mask = batch["attention_mask"]
        if is_ragged:
            token_embeds = select_token_hidden_state_layers(
                hidden_states,
                [0] if use_final_layer_fast_path else layers,
            )
            gather_targets = {
                "embeds": pad_for_gather(token_embeds, accelerator),
                "labels": pad_for_gather(labels, accelerator, label_ignore_index),
            }
            gathered = (
                gather_targets
                if not gather_across_processes
                or getattr(accelerator, "num_processes", 1) == 1
                else accelerator.gather_for_metrics(gather_targets)
            )
            token_embeds = gathered["embeds"].to(dtype=torch_dtype, device="cpu")
            # Keep the mask and the tensor it indexes on the same device.
            # Ragged embeddings are stored on CPU, so labels must move there
            # before filtering valid token positions as well.
            token_labels = gathered["labels"].to(device="cpu")
            valid = token_labels != label_ignore_index
            pooled = token_embeds[valid]
            labels = token_labels[valid]
        else:
            pooled = pool_hidden_state_layers(
                hidden_states,
                attention_mask,
                pooling,
                [0] if use_final_layer_fast_path else layers,
                normalize_before_pooling=normalize_before_pooling,
            )
            pooled_gather_targets: dict[str, Any] = {
                "pooled": pooled,
                "labels": labels,
            }
            if all_sequences is not None:
                pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0
                pooled_gather_targets["input_ids"] = pad_for_gather(
                    input_id_rows(batch, pad_token_id), accelerator, pad_token_id
                )
            gathered = (
                pooled_gather_targets
                if not gather_across_processes
                or getattr(accelerator, "num_processes", 1) == 1
                else accelerator.gather_for_metrics(pooled_gather_targets)
            )
            pooled = gathered["pooled"].to(dtype=torch_dtype, device="cpu")
            labels = gathered["labels"]
        if num_examples is not None:
            if embeddings is None:
                embeddings = torch.empty(
                    (num_examples, pooled.shape[-1]), dtype=torch_dtype
                )
            n = pooled.shape[0]
            embeddings[offset : offset + n] = pooled
            offset += n
        else:
            embeddings_chunks.append(pooled)
        all_labels.extend(labels.cpu().tolist())
        if all_sequences is not None:
            if tokenizer is None:
                raise ValueError("A tokenizer is required when decoding sequences.")
            all_sequences.extend(
                tokenizer.batch_decode(
                    gathered["input_ids"].cpu().tolist(), skip_special_tokens=True
                )
            )
        if ids is not None:
            has_ids = True
            if (
                gather_across_processes
                and getattr(accelerator, "num_processes", 1) != 1
            ):
                ids = accelerator.gather_for_metrics(ids, use_gather_object=True)
            if is_ragged:
                # One id per example; repeat each per its own valid-token
                # count so every flattened row still traces back to its
                # source example.
                counts = valid.sum(dim=1).tolist()
                all_ids.extend(
                    id_
                    for id_, count in zip(ids[: valid.shape[0]], counts)
                    for _ in range(count)
                )
            else:
                # Object gathers do not remove sampler padding, unlike tensor
                # gathers. The gathered embedding tensor is authoritative.
                all_ids.extend(ids[: pooled.shape[0]])

    if embeddings is None:
        embeddings = (
            torch.cat(embeddings_chunks, dim=0) if embeddings_chunks else torch.empty(0)
        )
    elif offset != num_examples:
        # Fewer rows were actually gathered than the dataset's nominal length
        # (e.g. a custom sampler/dataset without a 1:1 length correspondence);
        # trim the unused tail of the preallocated buffer rather than return
        # uninitialized memory.
        embeddings = embeddings[:offset]
    if not has_ids:
        all_ids = list(range(embeddings.shape[0]))
    return embeddings, all_ids, torch.tensor(all_labels), all_sequences


@dataclass(frozen=True)
class EmbeddingBackend:
    """Everything needed to re-embed the test split under a perturbed trunk
    or scrambled input, for the handful of control conditions
    (:mod:`~modules.evaluate.src.utils.controls`) that a pooled-embedding-only
    ``test_dataloader`` can no longer support once
    ``steps.prepare.embedding_cache.enabled=True`` swaps it in.

    This is the single place ``PreparedArtifacts`` exposes embedding-cache
    internals: ``TuneStep``/``ScoreStep`` never touch it, and ``PredictStep``
    only touches it to decide whether sequences need decoding separately
    (``token_test_dataloader``) -- every other embedding-cache-specific
    detail (trunk, pooling, layers, dtype, hidden size) stays behind this one
    field instead of five separate optional ``PreparedArtifacts`` attributes.
    """

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
    cache_policy: str = "reuse"
    model_id: str | None = None
    dataset_id: str | None = None
    max_length: int | None = None
    test_random_truncate: bool = False
    test_dataset_fingerprint: str | None = None

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
        )


class EmbeddingDataset(Dataset):
    """Wraps a cached ``(N, H)`` embedding tensor + aligned ids/labels."""

    def __init__(
        self, embeddings: torch.Tensor, ids: list[Any], labels: list[Any]
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
    return {
        "inputs_embeds": torch.stack([example["inputs_embeds"] for example in batch]),
        "labels": torch.tensor(
            [example["labels"] for example in batch], dtype=label_dtype
        ),
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
) -> dict[str, DataLoader | None]:
    """For each ``(dataloader, random_truncate)`` in *splits*, load that
    split's embeddings from the on-disk cache when reusable, otherwise
    extract them fresh from *model* (writing them to cache when eligible),
    then wrap them in an accelerator-prepared :class:`EmbeddingDataset`
    dataloader. Returns one dataloader (or ``None``, for an absent split)
    per key in *splits*, used by :class:`~modules.evaluate.src.steps.
    prepare.PrepareStep` to replace token dataloaders with pooled-embedding
    ones.

    ``is_ragged=True`` (token-level tasks) always skips the on-disk cache,
    regardless of ``cache_config.cache_policy``: per-token embeddings are
    far larger than pooled sequence embeddings, so they're only ever held
    in memory for the current run.

    Logging is explicitly gated on ``accelerator.is_main_process`` here
    (rather than relying on ``accelerate.logging.get_logger``'s automatic
    gating, as callers outside this module do) since this module's plain
    logger would otherwise print once per rank under multi-GPU runs.
    """
    embedding_dataloaders: dict[str, DataLoader | None] = {}
    cache_layers: Sequence[int] = (
        cache_config.layers if cache_config.layers is not None else (-1,)
    )
    # Computed once (not per split): identical for every split since it's a
    # property of the frozen trunk alone, not the data. Cheap relative to a
    # trunk forward pass, but still avoids repeating it three times.
    model_fingerprint = model_fingerprint or weights_fingerprint(model)
    for split, (dataloader, random_truncate) in splits.items():
        if dataloader is None:
            embedding_dataloaders[split] = None
            continue

        # Hub repo ids alone don't pin content (the default revision is
        # mutable), so fold in the trunk's actual weights and, when
        # available, the tokenized dataset's own content fingerprint --
        # otherwise a later push to the same model/dataset repo would
        # silently reuse embeddings computed from stale content.
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
        )
        # Ragged (token-level) embeddings are never persisted: they are far
        # larger than pooled sequence embeddings for the same dataset.
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
            extraction_dataloader = dataloader
            if extraction_batch_size is not None:
                extraction_dataloader = accelerator.prepare(
                    DataLoader(
                        dataloader.dataset,
                        batch_size=extraction_batch_size,
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
                autocast_dtype=_AUTOCAST_DTYPE[cache_config.autocast_dtype],
                normalize_before_pooling=cache_config.normalize_before_pooling,
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
                # Seeded generator so train shuffle order is reproducible.
                generator=(
                    torch.Generator().manual_seed(seed) if split == "train" else None
                ),
                collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
            )
        )
    return embedding_dataloaders
