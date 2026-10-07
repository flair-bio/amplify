"""Masked-marginal zero-shot variant-effect scoring.

For a deep-mutational-scan assay, the naive approach re-embeds the sequence
once per mutant. Masked-marginal scoring instead masks each *unique* mutated
position once, so the number of forward passes is bounded by the sequence
length rather than the (often much larger) number of scored variants --
e.g. a 300-residue assay with 6000 single-mutant variants needs at most 300
masked forward passes, not 6000.

Score for a (possibly multi-mutant) variant is the sum, over its mutated
positions, of ``log p(mutant aa | masked context) - log p(wildtype aa | masked
context)`` (Meier et al., 2021).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

_MUTATION_TOKEN_LEN = 3  # e.g. "A123T": wt aa, 1-indexed position, mutant aa


def parse_mutant(mutant: str) -> list[tuple[str, int, str]]:
    """Parse a (possibly ':'-joined multi-mutant) code into (wt_aa, pos, mut_aa) triples.

    Positions are 1-indexed into the wild-type sequence, per ProteinGym convention.
    """
    parsed = []
    for token in mutant.split(":"):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
        if match is None:
            raise ValueError(f"Invalid substitution notation: {token!r}")
        wt_aa, position, mut_aa = match.groups()
        pos = int(position)
        if pos < 1:
            raise ValueError(f"Invalid substitution notation: {token!r}")
        parsed.append((wt_aa, pos, mut_aa))
    return parsed


def _optimal_window(position: int, full_len: int, window: int) -> tuple[int, int]:
    """Token-index window of length ``window`` covering ``position``, centered
    where possible and clamped to sequence bounds otherwise.

    Matches the official ``compute_fitness.py``'s ``get_optimal_window``, so
    scoring a mutated position past ``window`` tokens from the sequence start
    sees the same surrounding context as the official ESM/ESM2 baselines
    instead of silently falling outside a fixed left-aligned truncation.
    """
    if full_len <= window:
        return 0, full_len
    half_window = window // 2
    if position < half_window:
        return 0, window
    if position >= full_len - half_window:
        return full_len - window, full_len
    return position - half_window, position + half_window


def _residue_token_indices(
    tokenizer: PreTrainedTokenizerBase, sequence: str
) -> list[int]:
    """Return model-input indices for the sequence's one-token amino acids."""
    full_ids = tokenizer(sequence, return_tensors="pt")["input_ids"][0].tolist()
    residue_ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0].tolist()
    if len(residue_ids) != len(sequence):
        raise ValueError(
            "ProteinGym substitution scoring requires one token per residue"
        )
    for start in range(len(full_ids) - len(residue_ids) + 1):
        if full_ids[start : start + len(residue_ids)] == residue_ids:
            return list(range(start, start + len(residue_ids)))
    raise ValueError("Could not locate residue tokens within the model input")


@torch.no_grad()
def score_assay_masked_marginal(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    wt_sequence: str,
    mutants: Sequence[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int | None = 1024,
) -> np.ndarray:
    """Return one masked-marginal score per entry in ``mutants`` (NaN if any of
    its mutated positions falls outside the wild-type sequence).

    Sequences longer than ``max_length`` tokens are never truncated outright:
    each masked position is instead scored inside a ``max_length``-token
    window centered on it (falling back to a left/right-aligned window at the
    sequence ends), matching the official ProteinGym ESM baselines' default
    ``scoring-window="optimal"`` behavior.
    """
    residue_indices = _residue_token_indices(tokenizer, wt_sequence)
    special_token_count = len(
        tokenizer(wt_sequence, return_tensors="pt")["input_ids"][0]
    ) - len(residue_indices)
    window = (
        len(wt_sequence) if max_length is None else max_length - special_token_count
    )
    if window < 1:
        raise ValueError(
            "max_length is too small to contain a residue and special tokens"
        )

    parsed_mutants = [parse_mutant(m) for m in mutants]
    positions = sorted({pos for muts in parsed_mutants for (_, pos, _) in muts})
    for mutations in parsed_mutants:
        for wt_aa, position, _ in mutations:
            if not 1 <= position <= len(wt_sequence):
                continue
            if wt_sequence[position - 1] != wt_aa:
                raise ValueError(
                    f"Wild-type residue mismatch at position {position}: expected "
                    f"{wt_sequence[position - 1]!r}, got {wt_aa!r}"
                )
    valid_positions = [p for p in positions if 1 <= p <= len(wt_sequence)]

    mask_id = tokenizer.mask_token_id
    log_probs_by_position: dict[int, torch.Tensor] = {}
    for start in range(0, len(valid_positions), batch_size):
        batch_positions = valid_positions[start : start + batch_size]
        windows = [
            _optimal_window(p - 1, len(wt_sequence), window) for p in batch_positions
        ]
        window_sequences = [wt_sequence[lo:hi] for lo, hi in windows]
        encoding = tokenizer(window_sequences, return_tensors="pt", padding=True)
        batch_input = encoding["input_ids"].to(device)
        local_indices = [
            _residue_token_indices(tokenizer, sequence)[p - 1 - lo]
            for p, (lo, _), sequence in zip(batch_positions, windows, window_sequences)
        ]
        for row, local_index in enumerate(local_indices):
            batch_input[row, local_index] = mask_id
        batch_attention = encoding.get("attention_mask")
        if batch_attention is not None:
            batch_attention = batch_attention.to(device)

        logits = model(input_ids=batch_input, attention_mask=batch_attention).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        for row, (position, local_index) in enumerate(
            zip(batch_positions, local_indices)
        ):
            log_probs_by_position[position] = log_probs[row, local_index].cpu()

    amino_acids = {
        aa
        for muts in parsed_mutants
        for (wt_aa, _, mut_aa) in muts
        for aa in (wt_aa, mut_aa)
    }
    aa_token_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in amino_acids}

    scores = np.full(len(mutants), np.nan, dtype=np.float64)
    for i, mutations in enumerate(parsed_mutants):
        total = 0.0
        for wt_aa, pos, mut_aa in mutations:
            log_probs = log_probs_by_position.get(pos)
            if log_probs is None:
                total = np.nan
                break
            total += (
                log_probs[aa_token_id[mut_aa]] - log_probs[aa_token_id[wt_aa]]
            ).item()
        scores[i] = total
    return scores


@torch.no_grad()
def _sequence_log_likelihood(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int,
    max_length: int | None,
) -> np.ndarray:
    """Sum of log p(token | context with that position masked), over every
    position: a true masked pseudo-log-likelihood (MPLL) sweep. Unlike a
    single unmasked forward pass, self-attention never sees a position's own
    token when that position is scored, at the cost of one forward pass per
    (sequence, position-batch) instead of one per sequence."""
    mask_id = tokenizer.mask_token_id
    scores = np.zeros(len(sequences), dtype=np.float64)
    for seq_index, sequence in enumerate(sequences):
        encoding = tokenizer(
            [sequence],
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        residue_indices = [
            position
            for position in _residue_token_indices(tokenizer, sequence)
            if position < input_ids.shape[1]
        ]

        total = 0.0
        for start in range(0, len(residue_indices), batch_size):
            positions = residue_indices[start : start + batch_size]
            batch_input = input_ids.repeat(len(positions), 1).clone()
            for row, position in enumerate(positions):
                batch_input[row, position] = mask_id
            batch_attention = (
                attention_mask.repeat(len(positions), 1)
                if attention_mask is not None
                else None
            )
            logits = model(input_ids=batch_input, attention_mask=batch_attention).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, position in enumerate(positions):
                total += log_probs[row, position, input_ids[0, position]].item()
        scores[seq_index] = total
    return scores


def score_assay_indel(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    wt_sequence: str,
    mutated_sequences: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
    max_length: int | None = 1024,
) -> np.ndarray:
    """Zero-shot score for indels: masked-marginal's position-alignment
    assumption breaks once mutant length differs from wild-type, so this
    scores each full sequence's masked pseudo-log-likelihood independently
    and takes the mutant-minus-wildtype difference."""
    log_likelihoods = _sequence_log_likelihood(
        model,
        tokenizer,
        [wt_sequence, *mutated_sequences],
        device,
        batch_size,
        max_length,
    )
    return log_likelihoods[1:] - log_likelihoods[0]
