"""Canonical amino-acid tokenization helpers."""

from __future__ import annotations

from typing import Any

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def resolve_aa_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Resolve distinct single-token IDs for the 20 canonical amino acids."""
    ids = tuple(tokenizer(CANONICAL_AMINO_ACIDS, add_special_tokens=False)["input_ids"])
    if (
        len(ids) != len(CANONICAL_AMINO_ACIDS)
        or len(set(ids)) != len(ids)
        or tokenizer.unk_token_id in ids
    ):
        raise ValueError(
            "The tokenizer must map each canonical amino acid to its own token."
        )
    return ids
