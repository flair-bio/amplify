"""Latent-alignment summed MLM and symmetric full-pool residue InfoNCE."""

import torch
from torch.nn import functional as F


def mlm_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != -100
    if not valid.any():
        return logits.reshape(-1)[:0].sum()
    kept = logits[valid]
    if kept.dtype in (torch.float16, torch.bfloat16):
        kept = kept.float()
    return F.cross_entropy(kept, labels[valid], reduction="sum")


def contrastive_chunk(
    sequence: torch.Tensor,
    structure: torch.Tensor,
    logit_scale: torch.Tensor,
    start: int,
    stop: int,
    *,
    min_inverse_temperature: float,
) -> torch.Tensor:
    """Both anchor directions classify against EVERY residue in the pool."""
    with torch.autocast(device_type=sequence.device.type, enabled=False):
        if sequence.dtype in (torch.float16, torch.bfloat16):
            sequence = sequence.float()
        structure = structure.to(sequence.dtype)
        scale = logit_scale.to(sequence.dtype).exp().clamp_min(min_inverse_temperature)
        target = torch.arange(start, stop, device=sequence.device)
        loss = (
            F.cross_entropy(
                scale * (sequence[start:stop] @ structure.T), target, reduction="sum"
            )
            + F.cross_entropy(
                scale * (structure[start:stop] @ sequence.T), target, reduction="sum"
            )
        ) / (2 * len(sequence))
    if not torch.isfinite(loss):
        raise FloatingPointError(
            "Nonfinite contrastive loss; inspect vectors and temperature."
        )
    return loss
