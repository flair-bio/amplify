import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from modules.post_train.src.config import AlignmentConfig, MLMConfig, TrainingConfig
from modules.post_train.src.model.alignment import AlignmentModel, forward
from modules.post_train.src.trainers.latent_alignment import (
    cached_backward, capture_rng, gradient_norms, make_optimizer, seed_everything, validation_mlm,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 12)
        self.dense = nn.Linear(12, 12)
        self.dropout = nn.Dropout(0.25)
        self.norm = nn.LayerNorm(12)
        self.last_residual = None

    def forward(self, ids):
        # Exercise all replayed RNG streams, not just Torch dropout.
        scale = 1 + (random.random() + np.random.random()) * 0.01 if self.training else 1
        residual = self.dense(self.dropout(self.embedding(ids))).tanh() * scale
        self.last_residual = residual
        return self.norm(residual)


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.amplify = TinyBackbone()
        self.lm_head = nn.Sequential(nn.Linear(12, 20), nn.GELU(), nn.Linear(20, 32))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.lm_head(self.amplify(input_ids)))


def toy_batch(length, no_mlm=False):
    ids = torch.arange(length + 2).unsqueeze(0) % 20 + 12
    ids[0, 0], ids[0, -1] = 3, 4
    residues = torch.ones_like(ids, dtype=torch.bool)
    residues[:, [0, -1]] = False
    labels = torch.full_like(ids, -100)
    if not no_mlm:
        labels[:, 1::2] = ids[:, 1::2]
        labels[~residues] = -100
    masked = ids.clone()
    masked[labels != -100] = 2
    return {
        "clean_input_ids": ids, "input_ids": masked, "residue_mask": residues,
        "mlm_labels": labels, "attention_mask": torch.ones_like(ids, dtype=torch.bool),
        "embeddings": torch.randn(length, 1024, dtype=torch.float64, requires_grad=True),
    }


def assert_rng_equal(expected, actual):
    assert expected[0] == actual[0]
    assert expected[1][0] == actual[1][0]
    np.testing.assert_array_equal(expected[1][1], actual[1][1])
    assert expected[1][2:] == actual[1][2:]
    assert torch.equal(expected[2], actual[2])


@pytest.mark.parametrize("empty_mlm,nondefault,pool_microbatches", [
    (False, False, 4),
    (False, True, 2),
    (True, True, 4),
])
def test_dense_vs_cached_loss_gradients_update_and_rng(empty_mlm, nondefault, pool_microbatches):
    seed_everything(29)
    alignment = AlignmentConfig(
        projection_dim=7, loss_weight=0.2, initial_temperature=0.2,
        min_inverse_temperature=8.0,
    ) if nondefault else AlignmentConfig()
    mlm_weight = 0.4 if nondefault else 1.0
    options = TrainingConfig(
        encoder_lr=2e-5, head_lr=2e-3, betas=(0.8, 0.9),
        weight_decay=0.03, gradient_clip=0.3,
    ) if nondefault else TrainingConfig()
    cached = AlignmentModel(TinyStudent(), alignment, teacher_dim=1024).double().train()
    direct = copy.deepcopy(cached)
    with torch.no_grad():
        direct.logit_scale.copy_(torch.tensor(math.log(1 / alignment.initial_temperature)))
    batches = [toy_batch(length, no_mlm=empty_mlm or i == 1)
               for i, length in enumerate((4, 7, 5, 9))]
    seed_everything(471)
    outputs = [forward(direct, batch) for batch in batches]
    count = sum(int((batch["mlm_labels"] != -100).sum()) for batch in batches)
    mlm = outputs[0]["mlm_logits"].sum() * 0
    for output, batch in zip(outputs, batches):
        kept = batch["mlm_labels"] != -100
        if kept.any():
            mlm = mlm + F.cross_entropy(output["mlm_logits"][kept], batch["mlm_labels"][kept], reduction="sum")
    mlm = mlm / max(1, count)
    total_residues = sum(len(output["sequence"]) for output in outputs)
    latent = mlm.new_zeros(())
    for begin in range(0, len(outputs), pool_microbatches):
        sequence = torch.cat([output["sequence"] for output in outputs[begin:begin + pool_microbatches]])
        structure = torch.cat([output["structure"] for output in outputs[begin:begin + pool_microbatches]])
        logits = direct.logit_scale.exp().clamp_min(alignment.min_inverse_temperature) * (sequence @ structure.T)
        target = torch.arange(len(sequence))
        latent = latent + (len(sequence) / total_residues) * (
            F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
    (mlm_weight * mlm + alignment.loss_weight * latent).backward()
    direct_rng = capture_rng(torch.device("cpu"))

    seed_everything(471)
    metrics = cached_backward(
        cached,
        batches,
        pool_microbatches,
        mlm_weight=mlm_weight,
        anchor_chunk_size=3,
        verify_replay=True,
    )
    assert_rng_equal(direct_rng, capture_rng(torch.device("cpu")))
    for (name, expected), (other, actual) in zip(direct.named_parameters(), cached.named_parameters()):
        assert name == other and expected.grad is not None and actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-11)
    assert all(batch["embeddings"].grad is None for batch in batches)
    assert metrics["mlm"] == pytest.approx(float(mlm.detach()), abs=1e-12)
    assert metrics["latent"] == pytest.approx(float(latent.detach()), abs=1e-12)
    assert metrics["total"] == pytest.approx(
        float((mlm_weight * mlm + alignment.loss_weight * latent).detach()), abs=1e-12
    )
    if not empty_mlm:
        assert all(value > 0 for role, value in gradient_norms(cached).items()
                   if not nondefault or role != "temperature")
    for model in (direct, cached):
        if model is direct:
            optimizer = torch.optim.AdamW([
                {"params": model.student.parameters(), "lr": options.encoder_lr},
                {"params": [p for name, p in model.named_parameters()
                            if not name.startswith("student.")], "lr": options.head_lr},
            ], betas=options.betas, weight_decay=options.weight_decay)
        else:
            optimizer = make_optimizer(model, options)
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        for parameter in model.parameters():
            parameter.grad.mul_(0.5)
        optimizer.step()
    for expected, actual in zip(direct.parameters(), cached.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)


def test_adapter_uses_pre_norm_and_removes_hook_on_error():
    model = AlignmentModel(TinyStudent(), teacher_dim=1024).double().eval()
    batch = toy_batch(4)
    result = model(batch)
    residual = model.student.amplify.last_residual[batch["residue_mask"]]
    expected = F.normalize(model.sequence_projection(residual), dim=-1)
    torch.testing.assert_close(result["sequence"], expected)
    assert not model.student.amplify.norm._forward_pre_hooks
    original = model.student.forward
    calls = 0

    def fail_clean(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("clean failed")
        return original(**kwargs)

    model.student.forward = fail_clean
    with pytest.raises(RuntimeError, match="clean failed"):
        model(batch)
    assert not model.student.amplify.norm._forward_pre_hooks


@pytest.mark.parametrize("mask_probability", [0.15, 1.0])
def test_validation_is_fixed_and_preserves_rng_and_mode(prepared_factory, vocabulary, mask_probability):
    manifest, _, _ = prepared_factory()
    from modules.post_train.src.dataset.manifest import load_manifest

    validation = load_manifest(manifest).validation
    model = AlignmentModel(TinyStudent(), teacher_dim=1024).train()
    mlm = MLMConfig(mask_probability=mask_probability)
    seed_everything(91)
    rng = capture_rng(torch.device("cpu"))
    first = validation_mlm(model, validation, 17, torch.device("cpu"), vocabulary=vocabulary, mlm=mlm)
    assert model.training
    assert_rng_equal(rng, capture_rng(torch.device("cpu")))
    assert first == validation_mlm(model, validation, 17, torch.device("cpu"), vocabulary=vocabulary, mlm=mlm)
    assert first["masked_residues"] > 0
    assert first["proteins"] == len(validation)
    if mask_probability == 1:
        assert first["masked_residues"] == sum(len(row.sequence) for row in validation)
        many_records = [validation[0]] * 33
        full_validation = validation_mlm(
            model, many_records, 17, torch.device("cpu"), vocabulary=vocabulary, mlm=mlm
        )
        assert full_validation["proteins"] == len(many_records)
        assert full_validation["masked_residues"] == sum(len(row.sequence) for row in many_records)
