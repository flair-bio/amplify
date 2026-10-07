"""
Symmetric low-rank bilinear contact-prediction head over a frozen/fine-tuned
trunk.

A dense ``[h_i, h_j, h_i*h_j, |h_i-h_j|]`` pair-feature tensor is
``O(L^2 * H)`` and prohibitively large (~GiB/sequence). Instead, project each
residue's hidden state to a small rank ``r`` and score every pair with a
single symmetric bilinear form -- ``O(L*r)`` feature memory, only the final
``(L, L)`` score matrix is ever materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, PretrainedConfig

from modules.evaluate.src.model.heads import _HeadOutput


@dataclass
class ContactModelConfig:
    """Single-argument constructor config for :class:`ContactPredictionModel`.

    Mirrors the one-argument ``type(model)(config)`` construction that
    ``utils.controls.ControlBuilder`` uses to build the
    ``random_trunk``/``random_head``/``random_both`` controls: it
    reconstructs a fresh instance from ``copy.deepcopy(model.config)`` alone,
    then selectively copies the trunk's or head's trained ``state_dict``
    back in.
    """

    trunk_config: PretrainedConfig
    rank: int = 128
    #: Positive-class weight for the BCE loss; contact maps are extremely
    #: imbalanced (few true contacts per (L, L) map), matching the legacy
    #: SequenceStructureEvaluator's ``contact_map_loss`` default.
    pos_weight: float = 30.0


class ContactPredictionModel(nn.Module):
    """Trunk + symmetric low-rank bilinear pairwise contact head.

    ``logits[b, i, j] = (W h_i) . (W h_j)``, i.e. the same projection ``W``
    scores both ends of the pair, so the map is symmetric by construction
    without an explicit ``(logits + logits.T) / 2`` step.
    """

    base_model_prefix = "trunk"

    def __init__(self, config: ContactModelConfig) -> None:
        super().__init__()
        self.config = config
        self.trunk = AutoModel.from_config(config.trunk_config)
        self.query = nn.Linear(config.trunk_config.hidden_size, config.rank)
        # Not a real class count (this is a per-pair binary score, not a
        # softmax head); kept only for compatibility with generic pipeline
        # code that reads a model's `num_labels` (e.g. control building).
        self.num_labels = 2

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _HeadOutput:
        hidden = self.trunk(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        z = self.query(hidden)  # (B, S, rank)
        logits = torch.einsum("bid,bjd->bij", z, z)  # (B, S, S), symmetric

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                pos_weight = torch.tensor(self.config.pos_weight, device=logits.device)
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask], labels[mask].to(logits.dtype), pos_weight=pos_weight
                )
            else:
                # No valid pair in this batch (e.g. every sequence has only
                # short-range pairs); a zero loss keeps the training loop's
                # backward()/optimizer.step() well-defined instead of
                # branching around it.
                loss = logits.sum() * 0.0
        return _HeadOutput(loss=loss, logits=logits)


def build_contact_model(
    model_id: str,
    rank: int = 128,
    pos_weight: float = 30.0,
    **from_pretrained_kwargs: Any,
) -> ContactPredictionModel:
    """Load a pretrained trunk with a freshly-initialized contact head."""
    trunk = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, **from_pretrained_kwargs
    )
    model = ContactPredictionModel(
        ContactModelConfig(trunk.config, rank=rank, pos_weight=pos_weight)
    )
    model.trunk = trunk
    return model
