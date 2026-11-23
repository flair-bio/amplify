"""Unit tests for the optional MLM training objectives."""

from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError
from torch.nn.functional import cross_entropy

from modules.pretrain.src.loss.loss import FocalLoss, LossConfig, get_loss


def _batch(seed: int = 0, vocab: int = 7):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, 6, vocab, generator=g)
    labels = torch.randint(0, vocab, (2, 6), generator=g)
    labels[0, :3] = -100
    return logits, labels


class TestFocalLoss:
    def test_gamma_zero_matches_model_cross_entropy(self):
        logits, labels = _batch()
        expected = cross_entropy(logits.view(-1, 7), labels.view(-1))
        torch.testing.assert_close(FocalLoss(0.0)(logits, labels), expected)

    def test_hand_computed_value(self):
        # Two supervised tokens with p_t = 1/2 and 1/4; one ignored token.
        logits = torch.log(
            torch.tensor([[[0.5, 0.5, 0.0], [0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]])
            + 1e-30
        )
        labels = torch.tensor([[0, 0, -100]])
        expected = (
            0.5**0.5 * math.log(2) + 0.75**0.5 * math.log(4)
        ) / 2  # (1 - p_t)^γ · (−log p_t), averaged over supervised tokens
        loss = FocalLoss(0.5)(logits, labels)
        assert loss.item() == pytest.approx(expected, rel=1e-5)

    def test_down_weights_confident_tokens(self):
        logits, labels = _batch()
        assert FocalLoss(0.5)(logits, labels) < FocalLoss(0.0)(logits, labels)

    def test_ignored_positions_do_not_contribute(self):
        logits, labels = _batch()
        perturbed = logits.clone()
        perturbed[labels == -100] = 100.0
        loss = FocalLoss(0.5)
        torch.testing.assert_close(loss(perturbed, labels), loss(logits, labels))

    def test_all_ignored_gives_zero_with_finite_gradient(self):
        logits = torch.randn(1, 4, 7, requires_grad=True)
        loss = FocalLoss(0.5)(logits, torch.full((1, 4), -100))
        loss.backward()
        assert loss.item() == 0.0
        assert torch.isfinite(logits.grad).all()

    def test_bf16_logits_give_fp32_loss_and_finite_grad_at_zero_ce(self):
        logits, labels = _batch()
        # One token predicted with certainty: CE = 0, where (1 - p_t)^0.5 is not
        # differentiable without the clamp floor.
        logits[1, 0] = -1e4
        logits[1, 0, labels[1, 0]] = 1e4
        logits = logits.bfloat16().requires_grad_()
        loss = FocalLoss(0.5)(logits, labels)
        loss.backward()
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        assert torch.isfinite(logits.grad).all()

    def test_packed_shape(self):
        logits, labels = _batch()
        packed = FocalLoss(0.5)(logits.reshape(1, 12, 7), labels.reshape(1, 12))
        torch.testing.assert_close(packed, FocalLoss(0.5)(logits, labels))


class TestLossConfig:
    def test_cross_entropy_uses_model_loss(self):
        assert LossConfig().type == "cross_entropy"
        assert get_loss(LossConfig()) is None

    def test_focal_dispatch(self):
        loss = get_loss(LossConfig(type="focal", gamma=0.5))
        assert isinstance(loss, FocalLoss) and loss.gamma == 0.5

    @pytest.mark.parametrize(
        "kwargs", [{"gamma": -0.1}, {"gamma": float("inf")}, {"averaging": "token"}]
    )
    def test_rejects_invalid(self, kwargs):
        with pytest.raises(ValidationError):
            LossConfig(type="focal", **kwargs)
