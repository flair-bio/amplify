"""Residue-preserving tokenization and configurable MLM corruption."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from modules.post_train.src.config import MLMConfig
from modules.post_train.src.dataset.manifest import (
    Record,
    load_embeddings,
)
from modules.post_train.src.model.tokenization import Vocabulary

Batch = dict[str, torch.Tensor]


def collate_records(
    records: Sequence[Record],
    generator: torch.Generator,
    device: torch.device | str,
    *,
    vocabulary: Vocabulary,
    mlm: MLMConfig | None = None,
) -> Batch:
    mlm = mlm if mlm is not None else MLMConfig()
    if not records:
        raise ValueError("Cannot collate an empty microbatch.")
    width = max(len(row.sequence) for row in records) + 2
    clean = torch.full((len(records), width), vocabulary.pad_id, dtype=torch.long)
    attention = torch.zeros_like(clean, dtype=torch.bool)
    residues = torch.zeros_like(clean, dtype=torch.bool)
    vectors = []
    for index, row in enumerate(records):
        length = len(row.sequence)
        clean[index, : length + 2] = torch.tensor(vocabulary.encode(row.sequence))
        attention[index, : length + 2] = True
        residues[index, 1 : length + 1] = True
        vectors.append(torch.from_numpy(load_embeddings(row)))
    masked, labels = clean.clone(), torch.full_like(clean, -100)
    chosen = (
        torch.rand(clean.shape, generator=generator) < mlm.mask_probability
    ) & residues
    labels[chosen] = clean[chosen]
    replace = (
        torch.rand(clean.shape, generator=generator) < mlm.mask_replace_probability
    ) & chosen
    conditional_random = (
        mlm.random_replace_probability / (1 - mlm.mask_replace_probability)
        if mlm.mask_replace_probability < 1
        else 0.0
    )
    random_replace = (
        (torch.rand(clean.shape, generator=generator) < conditional_random)
        & chosen
        & ~replace
    )
    random_ids = torch.randint(vocabulary.size, clean.shape, generator=generator)
    masked[replace] = vocabulary.mask_id
    masked[random_replace] = random_ids[random_replace]
    # Attention comes from the clean sequence: a random PAD is still a residue.
    return {
        key: value.to(device)
        for key, value in {
            "clean_input_ids": clean,
            "input_ids": masked,
            "attention_mask": attention,
            "residue_mask": residues,
            "mlm_labels": labels,
            "embeddings": torch.cat(vectors),
        }.items()
    }
