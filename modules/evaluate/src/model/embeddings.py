"""Frozen-trunk extraction, hidden-state selection, and residue pooling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.evaluate.src.dataset.collator import input_id_rows, unpack_packed
from modules.evaluate.src.model.inference import pad_for_gather

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import EmbeddingDtype, EmbeddingPooling

_TORCH_DTYPE = {"float16": torch.float16, "float32": torch.float32}
AUTOCAST_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def validate_embedding_cache_compat(
    is_ragged: bool, tokenizer: Any, pooling: EmbeddingPooling
) -> None:
    """Reject CLS pooling unless position zero is a fixed special token."""
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


def residue_attention_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, token_config: Any
) -> torch.Tensor:
    """Exclude boundary and padding tokens without masking unknown residues."""
    mask = attention_mask.bool()
    for name in (
        "cls_token_id",
        "bos_token_id",
        "sep_token_id",
        "eos_token_id",
        "pad_token_id",
    ):
        token_id = getattr(token_config, name, None)
        if isinstance(token_id, int):
            mask = mask & (input_ids != token_id)
    return mask


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: EmbeddingPooling,
    normalize_before_pooling: bool = False,
) -> torch.Tensor:
    """Pool ``(B, S, H)`` states to ``(B, H)`` in float32.

    Mean pooling uses the supplied residue mask; CLS selects position zero.
    Optional L2 normalization applies independently to each token.
    """
    hidden_states = hidden_states.float()
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
    """Pool layers independently and concatenate in requested layer order."""
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
    """Concatenate selected layers feature-wise without sequence pooling."""
    selected = _select_layers(hidden_states, layers)
    return torch.cat(selected, dim=-1)


def autocast_context(
    device: torch.device | None, autocast_dtype: torch.dtype | None
) -> Any:
    """Return a safe autocast context for the device."""
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
    progress_label: str | None = None,
) -> tuple[torch.Tensor, list[Any], torch.Tensor, list[str] | None]:
    """Extract aligned CPU embeddings, IDs, labels, and optional decoded inputs.

    Sequence tasks pool each layer into one row per example. Ragged tasks keep
    one row per valid label token and repeat the source ID for each row.
    Labels are captured in the same pass to preserve shuffled-loader alignment.
    The final-layer fast path does not request the complete hidden-state tuple.
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
    example_offset = 0
    all_labels: list[Any] = []
    all_sequences: list[str] | None = [] if tokenizer is not None else None
    has_ids = False
    trunk = getattr(model, "base_model", model)
    trunk.eval()
    pooling_tokenizer = tokenizer or getattr(
        getattr(dataloader, "collate_fn", None),
        "tokenizer",
        getattr(trunk, "config", None),
    )

    # Preallocate pooled rows to avoid doubling host RAM at the final cat.
    try:
        num_examples: int | None = (
            None if is_ragged else len(dataloader.dataset)  # type: ignore[arg-type]
        )
    except (AttributeError, TypeError):
        num_examples = None
    embeddings: torch.Tensor | None = None
    embeddings_chunks: list[torch.Tensor] = []
    offset = 0
    progress = None
    if progress_label and accelerator.is_main_process:
        try:
            total_batches = len(dataloader)
        except (AttributeError, TypeError, ValueError):
            total_batches = None
        progress = tqdm(
            dataloader,
            desc=progress_label,
            total=total_batches,
            unit="batch",
            mininterval=1.0,
            dynamic_ncols=True,
        )
        batches = progress
    else:
        batches = dataloader
    for batch_idx, batch in enumerate(batches, 1):
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
        with autocast_context(getattr(accelerator, "device", None), autocast_dtype):
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
            token_labels = gathered["labels"].to(device="cpu")
            valid = token_labels != label_ignore_index
            pooled = token_embeds[valid]
            labels = token_labels[valid]
        else:
            attention_mask = residue_attention_mask(
                input_id_rows(batch, 0), attention_mask, pooling_tokenizer
            )
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
        if ids is None and is_ragged:
            ids = list(range(example_offset, example_offset + valid.shape[0]))
        if ids is not None:
            has_ids = True
            if (
                gather_across_processes
                and getattr(accelerator, "num_processes", 1) != 1
            ):
                ids = accelerator.gather_for_metrics(ids, use_gather_object=True)
            if is_ragged:
                counts = valid.sum(dim=1).tolist()
                all_ids.extend(
                    id_
                    for id_, count in zip(ids[: valid.shape[0]], counts)
                    for _ in range(count)
                )
            else:
                # Object gathers do not remove sampler padding.
                all_ids.extend(ids[: pooled.shape[0]])
        if is_ragged:
            example_offset += valid.shape[0]
        if progress is not None:
            progress.set_postfix(rows=len(all_labels), refresh=False)

    if embeddings is None:
        embeddings = (
            torch.cat(embeddings_chunks, dim=0) if embeddings_chunks else torch.empty(0)
        )
    elif offset != num_examples:
        embeddings = embeddings[:offset]
    if not has_ids:
        all_ids = list(range(embeddings.shape[0]))
    return embeddings, all_ids, torch.tensor(all_labels), all_sequences
