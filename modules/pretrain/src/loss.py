"""Optional training objectives for masked amino-acid prediction."""

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.nn.functional import cross_entropy


class LossConfig(BaseModel):
    """MLM training objective; ``cross_entropy`` keeps the model's built-in loss.

    Attributes:
        type: ``"cross_entropy"`` or ``"focal"`` (Lin et al., 2017).
        gamma: Focal focusing exponent; ``0`` recovers cross-entropy.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["cross_entropy", "focal"] = "cross_entropy"
    gamma: float = Field(0.5, ge=0.0, allow_inf_nan=False)


class FocalLoss(nn.Module):
    """Token-averaged focal loss ``(1 - p_t)^gamma * CE`` over labels != -100."""

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # FP32: the trainer calls this outside the model's autocast region.
        ce = cross_entropy(
            logits.float().flatten(0, -2),
            labels.flatten(),
            ignore_index=-100,
            reduction="none",
        )
        # -expm1(-ce) = 1 - p_t, accurate near p_t = 1; the floor keeps
        # fractional-gamma gradients finite where ce = 0.
        modulation = (-torch.expm1(-ce)).clamp_min(torch.finfo(ce.dtype).tiny)
        loss = modulation.pow(self.gamma) * ce
        return loss.sum() / labels.ne(-100).sum().clamp_min(1)


def get_loss(config: LossConfig) -> nn.Module | None:
    """Return a ``Trainer.loss_fn``, or ``None`` to use the model's cross-entropy."""
    if config.type == "focal":
        return FocalLoss(config.gamma)
    return None
