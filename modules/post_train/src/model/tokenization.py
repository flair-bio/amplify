"""Discover residue and special-token IDs from checkpoint-owned assets."""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    pad_id: int
    mask_id: int
    bos_id: int
    eos_id: int

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: str) -> list[int]:
        try:
            return [
                self.bos_id,
                *[self.token_to_id[aa] for aa in sequence],
                self.eos_id,
            ]
        except KeyError as error:
            raise ValueError(
                f"Student vocabulary does not contain residue {error.args[0]!r}."
            ) from error


def load_vocabulary(
    checkpoint: Path,
    *,
    residues: Collection[str] | None = None,
) -> Vocabulary:
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    if tokenizer.truncation is not None:
        raise ValueError("Student tokenizer must not truncate residue sequences.")
    config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    tokens = tokenizer.get_vocab()
    if sorted(tokens.values()) != list(range(len(tokens))):
        raise ValueError("Student vocabulary must have unique, contiguous token IDs.")
    special = {}
    for role in ("pad", "mask", "bos", "eos"):
        token = config.get(f"{role}_token")
        if isinstance(token, dict):
            token = token.get("content")
        if not isinstance(token, str) or token not in tokens:
            raise ValueError(f"Student tokenizer must declare a valid {role}_token.")
        special[f"{role}_id"] = tokens[token]
    if len(set(special.values())) != len(special):
        raise ValueError("Student PAD, MASK, BOS and EOS token IDs must be distinct.")
    vocabulary = Vocabulary(token_to_id=tokens, **special)
    alphabet = "".join(
        sorted(
            residues
            if residues is not None
            else (token for token in tokens if len(token) == 1 and "A" <= token <= "Z")
        )
    )
    if not alphabet:
        raise ValueError("Student tokenizer must support residue tokens.")
    for sequence in (alphabet, alphabet[::-1], alphabet * 2):
        if tokenizer.encode(sequence).ids != vocabulary.encode(sequence):
            raise ValueError(
                "Tokenizer must preserve one token per residue with BOS/EOS, without truncation."
            )
    return vocabulary
