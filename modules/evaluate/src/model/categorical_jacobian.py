from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, PretrainedConfig

from modules.evaluate.src.model.heads import _HeadOutput

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MAX_JACOBIAN_BYTES = 512 * 1024 * 1024


@dataclass
class CategoricalJacobianConfig:
    """Single-argument config carrying the trunk architecture and Jacobian
    scoring parameters needed to construct/serialize a
    :class:`CategoricalJacobianModel`."""

    trunk_config: PretrainedConfig
    aa_token_ids: tuple[int, ...]
    mutant_batch_size: int = 256
    max_jacobian_bytes: int = DEFAULT_MAX_JACOBIAN_BYTES
    leading_token_id: int | None = None
    trailing_token_id: int | None = None


def resolve_aa_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Resolve canonical amino-acid token ids for the active tokenizer."""
    # A single round-trip encoding catches both subword-splitting (wrong
    # token count) and degenerate vocabularies (repeated/unk ids) at once.
    ids = tuple(tokenizer(CANONICAL_AMINO_ACIDS, add_special_tokens=False)["input_ids"])
    if len(ids) != len(CANONICAL_AMINO_ACIDS) or len(set(ids)) != len(ids):
        raise ValueError(
            "Categorical Jacobian requires the tokenizer to map each canonical "
            "amino acid to its own distinct single token."
        )
    return ids


def resolve_special_token_layout(tokenizer: Any) -> tuple[int, int]:
    """Require the BOS/EOS layout assumed by the Jacobian coordinate system."""
    # Identify the start-of-sequence token
    leading = next(
        (
            token_id
            for token_id in (tokenizer.bos_token_id, tokenizer.cls_token_id)
            if token_id is not None
        ),
        None,
    )
    # Identify the end-of-sequence token
    trailing = next(
        (
            token_id
            for token_id in (tokenizer.eos_token_id, tokenizer.sep_token_id)
            if token_id is not None
        ),
        None,
    )

    # The Jacobian indexing relies on exactly one special token at each end
    if leading is None or trailing is None:
        raise ValueError(
            "Categorical Jacobian requires exactly one leading and trailing "
            "special token for residue-label alignment."
        )
    return int(leading), int(trailing)


def jacobian_to_contacts(jacobian: torch.Tensor) -> torch.Tensor:
    """Convert a ``(L,20,L,20)`` Jacobian tensor into an ``(L,L)`` contact map."""
    centered = jacobian.to(torch.float32)

    # A single left-to-right sweep over the 4 axes already reaches the
    # zero-mean fixed point for every axis: centering axis k only ever
    # subtracts a value constant along k, so it cannot reintroduce a nonzero
    # mean on an already-centered axis (its own mean over that axis is 0).
    # No repeated sweeps needed.
    for axis in range(4):
        centered = centered - centered.mean(dim=axis, keepdim=True)

    # Directional Jacobian entries must be symmetrized before the nonlinear
    # Frobenius reduction: (j, b, i, a) and (i, a, j, b) describe the same
    # residue pair from opposite directions.
    centered = 0.5 * (centered + centered.permute(2, 3, 0, 1))
    pair_scores = torch.linalg.vector_norm(centered, dim=(1, 3))

    # The diagonal (i == j) is the trivially large self-sensitivity term, not
    # a residue pair -- exclude it from the APC statistics or it dominates
    # every row/col/global mean it touches.
    length = pair_scores.shape[0]
    off_diag = ~torch.eye(length, dtype=torch.bool, device=pair_scores.device)
    masked_scores = pair_scores * off_diag
    denom = max(length - 1, 1)
    row_mean = masked_scores.sum(dim=1, keepdim=True) / denom
    col_mean = masked_scores.sum(dim=0, keepdim=True) / denom
    global_mean = (masked_scores.sum() / max(length * denom, 1)).clamp_min(1e-8)
    apc = row_mean * col_mean / global_mean

    return (pair_scores - apc) * off_diag


class CategoricalJacobianModel(nn.Module):
    """Masked-LM trunk wrapper emitting pairwise zero-shot contact scores."""

    base_model_prefix = "trunk"

    def __init__(
        self, config: CategoricalJacobianConfig, trunk: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        # Accept a pre-built (e.g. pretrained) trunk so callers that already
        # have one don't pay for a throwaway randomly-initialized instance.
        self.trunk = (
            trunk
            if trunk is not None
            else AutoModelForMaskedLM.from_config(config.trunk_config)
        )
        self.register_buffer(
            "aa_token_ids", torch.tensor(config.aa_token_ids, dtype=torch.long)
        )
        self.mutant_batch_size = int(config.mutant_batch_size)
        self.max_jacobian_bytes = int(config.max_jacobian_bytes)

        if self.mutant_batch_size <= 0:
            raise ValueError("mutant_batch_size must be greater than zero.")
        if self.max_jacobian_bytes <= 0:
            raise ValueError("max_jacobian_bytes must be greater than zero.")
        self.num_labels = 2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _HeadOutput:
        del kwargs
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        logits = torch.empty(
            batch_size, seq_len, seq_len, device=device, dtype=torch.float32
        )
        aa_count = int(self.aa_token_ids.numel())

        if (
            self.config.leading_token_id is not None
            or self.config.trailing_token_id is not None
        ):
            lengths = attention_mask.sum(dim=1)
            has_tokens = lengths > 0
            boundary_ok = torch.ones(batch_size, dtype=torch.bool, device=device)
            if self.config.leading_token_id is not None:
                boundary_ok &= input_ids[:, 0] == self.config.leading_token_id
            if self.config.trailing_token_id is not None:
                trailing_positions = (lengths - 1).clamp_min(0)
                trailing_tokens = input_ids.gather(1, trailing_positions.unsqueeze(1))
                boundary_ok &= (
                    trailing_tokens.squeeze(1) == self.config.trailing_token_id
                )
            if not torch.all(boundary_ok | ~has_tokens).item():
                raise ValueError(
                    "Categorical Jacobian input does not match the expected "
                    "leading/trailing special token layout."
                )

        # Guards against building a mutation-loop autograd graph when this
        # model is invoked outside the eval pipeline's own inference_mode().
        with torch.no_grad():
            # Process one sequence at a time to manage the large Jacobian allocations
            for batch_index in range(batch_size):
                length = int(attention_mask[batch_index].sum().item())

                # Skip empty or special-token-only sequences
                if length <= 2:
                    logits[batch_index] = torch.zeros(
                        (seq_len, seq_len), device=device, dtype=torch.float32
                    )
                    continue

                sequence = input_ids[batch_index, :length]
                num_positions = length - 2  # Exclude BOS/EOS tokens
                total_mutants = num_positions * aa_count

                # Fall back to a half-precision resident buffer before
                # aborting on sequences that don't fit at full precision.
                jacobian_dtype = torch.float32
                jacobian_bytes = num_positions * aa_count * num_positions * aa_count * 4
                if jacobian_bytes > self.max_jacobian_bytes:
                    jacobian_dtype = torch.float16
                    jacobian_bytes //= 2
                if jacobian_bytes > self.max_jacobian_bytes:
                    raise ValueError(
                        "Jacobian tensor would require "
                        f"{jacobian_bytes / 1024**2:.1f} MiB even at half "
                        "precision, exceeding the configured limit of "
                        f"{self.max_jacobian_bytes / 1024**2:.1f} MiB."
                    )

                # (L_pos, 20, L_pos, 20) tensor viewed as (L_pos * 20, L_pos, 20) for fast slicing
                jacobian = torch.zeros(
                    (num_positions, aa_count, num_positions, aa_count),
                    device=device,
                    dtype=jacobian_dtype,
                )
                jacobian_flat = jacobian.view(-1, num_positions, aa_count)

                seq_attn = attention_mask[batch_index, :length]

                # Single wild-type baseline: the Jacobian entry for mutation
                # (j, b) is the *change* it causes at (i, a), not the raw
                # mutant logit, so every chunk below subtracts this same
                # reference (computed once, not once per mutation).
                baseline_logits = (
                    self.trunk(
                        input_ids=sequence.unsqueeze(0),
                        attention_mask=seq_attn.unsqueeze(0),
                    )
                    .logits[0, 1 : length - 1, self.aa_token_ids]
                    .to(torch.float32)
                )

                # Iterate over all possible single-point mutations in VRAM-safe batches
                for start in range(0, total_mutants, self.mutant_batch_size):
                    end = min(start + self.mutant_batch_size, total_mutants)
                    chunk_size = end - start

                    # Map 1D index back to sequence position and amino acid token
                    chunk_indices = torch.arange(start, end, device=device)
                    chunk_pos = (
                        chunk_indices // aa_count
                    ) + 1  # Shifted by 1 to skip BOS token
                    chunk_aa = self.aa_token_ids[chunk_indices % aa_count]

                    # Create the mutated sequences for this chunk
                    chunk_inputs = sequence.unsqueeze(0).expand(chunk_size, -1).clone()
                    chunk_inputs[torch.arange(chunk_size, device=device), chunk_pos] = (
                        chunk_aa
                    )
                    chunk_attn = seq_attn.unsqueeze(0).expand(chunk_size, -1)

                    # Forward pass, converted to a delta against the WT baseline
                    chunk_logits = (
                        self.trunk(
                            input_ids=chunk_inputs,
                            attention_mask=chunk_attn,
                        )
                        .logits[:, 1 : length - 1, self.aa_token_ids]
                        .to(torch.float32)
                        - baseline_logits
                    )

                    # Direct contiguous tensor copy into flat view
                    jacobian_flat[start:end] = chunk_logits

                # Collapse the 4D mutational sensitivity tensor into a 2D spatial contact map
                contact_map = jacobian_to_contacts(jacobian)
                map_min = float(contact_map.min().item())

                # Embed the variable-length contact map back into the padded batch shape
                full_map = torch.full(
                    (seq_len, seq_len), map_min, device=device, dtype=torch.float32
                )
                full_map[1 : length - 1, 1 : length - 1] = contact_map

                logits[batch_index] = full_map

        # No loss: this is a frozen zero-shot scorer, not a trained
        # classifier, so its Frobenius/APC scores aren't calibrated
        # log-odds (`labels` is accepted only for interface parity with
        # other models and is otherwise unused).
        del labels
        return _HeadOutput(loss=None, logits=logits)


def build_categorical_jacobian_model(
    model_id: str,
    mutant_batch_size: int = 256,
    max_jacobian_bytes: int = DEFAULT_MAX_JACOBIAN_BYTES,
    **from_pretrained_kwargs: Any,
) -> CategoricalJacobianModel:
    """Load a pretrained masked-LM trunk for Jacobian contact scoring."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Establish and validate tokenizer mapping constraints required for the Jacobian
    aa_token_ids = resolve_aa_token_ids(tokenizer)
    leading_token_id, trailing_token_id = resolve_special_token_layout(tokenizer)

    trunk = AutoModelForMaskedLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        **from_pretrained_kwargs,
    )

    # Wrap the standard MLM into the specialized contact-prediction model.
    # Pass the already-loaded pretrained trunk directly so __init__ doesn't
    # also allocate a throwaway randomly-initialized one from trunk.config.
    return CategoricalJacobianModel(
        CategoricalJacobianConfig(
            trunk_config=trunk.config,
            aa_token_ids=aa_token_ids,
            mutant_batch_size=mutant_batch_size,
            max_jacobian_bytes=max_jacobian_bytes,
            leading_token_id=leading_token_id,
            trailing_token_id=trailing_token_id,
        ),
        trunk=trunk,
    )
