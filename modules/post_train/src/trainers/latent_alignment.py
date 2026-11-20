"""Latent-alignment optimization, gradient-cache replay and MLM diagnostics."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch

from modules.post_train.src.contracts.config import MLMConfig, TrainingConfig
from modules.post_train.src.dataset.collator import Batch, collate_records
from modules.post_train.src.dataset.latent_alignment import Record
from modules.post_train.src.model.alignment import AlignmentModel, forward
from modules.post_train.src.model.tokenization import Vocabulary
from modules.post_train.src.trainers.losses import contrastive_chunk, mlm_sum


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng(device: torch.device) -> tuple[Any, ...]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    )


def restore_rng(state: tuple[Any, ...], device: torch.device) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state(state[3], device)


def cached_backward(
    model: AlignmentModel,
    batches: list[Batch],
    pool_microbatches: int,
    *,
    anchor_chunk_size: int = 256,
    verify_replay: bool = False,
) -> dict[str, float | int]:
    """Accumulate one update's gradients; caller zeroes, clips and steps."""
    if (
        pool_microbatches < 1
        or anchor_chunk_size < 1
        or not batches
        or len(batches) % pool_microbatches
    ):
        raise ValueError("Require complete contrastive pools and positive chunk sizes.")
    device = batches[0]["input_ids"].device
    states, sequence_parts, structure_parts, sizes = [], [], [], []
    masked_count = sum(int((batch["mlm_labels"] != -100).sum()) for batch in batches)
    with torch.no_grad():
        for batch in batches:
            states.append(capture_rng(device))
            output = forward(model, batch)
            sequence_parts.append(output["sequence"].detach())
            structure_parts.append(output["structure"].detach())
            sizes.append(len(output["sequence"]))
            del output
    after_cache = capture_rng(device)
    sequence, structure = torch.cat(sequence_parts), torch.cat(structure_parts)
    if sequence.dtype in (torch.float16, torch.bfloat16):
        sequence, structure = sequence.float(), structure.float()
    sequence = sequence.detach().requires_grad_(True)
    structure = structure.detach().requires_grad_(True)
    del sequence_parts, structure_parts
    total_residues = sum(sizes)
    latent_value, mlm_value, offset = 0.0, 0.0, 0
    for begin in range(0, len(batches), pool_microbatches):
        pool_size = sum(sizes[begin : begin + pool_microbatches])
        if pool_size < 2:
            raise ValueError("A contrastive pool needs at least two residues.")
        for start in range(0, pool_size, anchor_chunk_size):
            loss = contrastive_chunk(
                sequence[offset : offset + pool_size],
                structure[offset : offset + pool_size],
                model.logit_scale,
                start,
                min(start + anchor_chunk_size, pool_size),
                min_inverse_temperature=model.alignment.min_inverse_temperature,
            ) * (pool_size / total_residues)
            (model.alignment.loss_weight * loss).backward()
            latent_value += float(loss.detach())
        offset += pool_size
    sequence_gradient, structure_gradient = sequence.grad, structure.grad
    assert sequence_gradient is not None and structure_gradient is not None
    offset = 0
    try:
        for state, batch, size in zip(states, batches, sizes, strict=True):
            restore_rng(state, device)
            output = forward(model, batch)
            if verify_replay:
                for key, expected in (("sequence", sequence), ("structure", structure)):
                    torch.testing.assert_close(
                        output[key].to(expected.dtype),
                        expected[offset : offset + size],
                        rtol=1e-5,
                        atol=1e-6,
                    )
            surrogate = (
                output["sequence"].to(sequence_gradient.dtype)
                * sequence_gradient[offset : offset + size]
            ).sum()
            surrogate += (
                output["structure"].to(structure_gradient.dtype)
                * structure_gradient[offset : offset + size]
            ).sum()
            mlm = mlm_sum(output["mlm_logits"], batch["mlm_labels"]) / max(
                1, masked_count
            )
            (surrogate + mlm).backward()
            mlm_value += float(mlm.detach())
            offset += size
            del output, surrogate, mlm
    finally:
        restore_rng(after_cache, device)
    return {
        "mlm": mlm_value,
        "latent": latent_value,
        "total": mlm_value + model.alignment.loss_weight * latent_value,
        "masked_residues": masked_count,
        "aligned_residues": total_residues,
    }


def make_optimizer(model: AlignmentModel, options: TrainingConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [
            {"params": model.student.parameters(), "lr": options.encoder_lr},
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("student.")
                ],
                "lr": options.head_lr,
            },
        ],
        betas=options.betas,
        weight_decay=options.weight_decay,
    )


def gradient_norms(model: AlignmentModel) -> dict[str, float]:
    groups: dict[str, list[torch.Tensor]] = {
        "encoder": [],
        "mlm_decoder": [],
        "projections": [],
        "temperature": [],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            role = (
                "encoder"
                if name.startswith("student.amplify.")
                else "mlm_decoder"
                if name.startswith("student.lm_head.")
                else "temperature"
                if name == "logit_scale"
                else "projections"
            )
            groups[role].append(parameter.grad.detach().float().square().sum())
    return {
        role: float(torch.stack(values).sum().sqrt()) if values else 0.0
        for role, values in groups.items()
    }


@torch.no_grad()
def validation_mlm(
    model: AlignmentModel,
    records: list[Record],
    seed: int,
    device: torch.device,
    *,
    vocabulary: Vocabulary,
    mlm: MLMConfig | None = None,
) -> dict[str, Any]:
    was_training = model.training
    rng = capture_rng(device)
    model.eval()
    selected = []
    for source in sorted({row.source for row in records}):
        group = [row for row in records if row.source == source]
        selected.extend(
            sorted(
                group,
                key=lambda row: hashlib.sha256(
                    (str(seed) + row.id).encode()
                ).hexdigest(),
            )[:32]
        )
    per_source: dict[str, list[float]] = {}
    try:
        for row in selected:
            record_seed = int.from_bytes(
                hashlib.sha256((str(seed) + row.id).encode()).digest()[:8], "little"
            )
            batch = collate_records(
                [row],
                torch.Generator().manual_seed(record_seed),
                device,
                vocabulary=vocabulary,
                mlm=mlm,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model.student(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
            entry = per_source.setdefault(row.source, [0.0, 0.0])
            entry[0] += float(mlm_sum(logits, batch["mlm_labels"]))
            entry[1] += int((batch["mlm_labels"] != -100).sum())
    finally:
        model.train(was_training)
        restore_rng(rng, device)
    if not per_source or any(count == 0 for _, count in per_source.values()):
        raise ValueError("Fixed validation masks selected no residues for a source.")
    count = int(sum(entry[1] for entry in per_source.values()))
    source_means = {
        source: nll / number for source, (nll, number) in per_source.items()
    }
    return {
        "mlm": sum(entry[0] for entry in per_source.values()) / count,
        "masked_residues": count,
        "proteins": len(selected),
        "sources": source_means,
        "balanced_mlm": sum(source_means.values()) / len(source_means),
    }
