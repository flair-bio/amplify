"""Tests for the zero-shot CASP categorical-Jacobian predictor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import EsmForMaskedLM

from modules.evaluate.src.model.adapters import masked_lm_logits
from modules.evaluate.src.model.contact_jacobian import ContactJacobianModel, _canonical_log_probabilities, jacobian_to_contacts
from modules.evaluate.src.dataset.contact import ContactChain, align_contact_sequence
from modules.evaluate.src.utils.amino_acids import CANONICAL_AMINO_ACIDS
from modules.pretrain.src.model.modeling_amplify import AMPLIFYForMaskedLM


class _Tokenizer:
    bos_token_id = 0
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        self.ids = {aa: i + 4 for i, aa in enumerate(CANONICAL_AMINO_ACIDS)}

    def __call__(self, sequence: str, **kwargs: object) -> dict[str, list[int]]:
        add_special_tokens = kwargs.get("add_special_tokens", True)
        input_ids = [self.ids[aa] for aa in sequence]
        if not add_special_tokens:
            return {"input_ids": input_ids}
        return {
            "input_ids": [self.bos_token_id] + input_ids + [self.eos_token_id],
            "special_tokens_mask": [1] + [0] * len(sequence) + [1],
        }

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.ids[token] for token in tokens]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_lm_composition_preserves_logits_and_head_method(model_family, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    kwargs = dict(model_family.config_kwargs)
    config_cls = kwargs.pop("config_cls")
    model_cls = (
        AMPLIFYForMaskedLM if model_family.name == "amplify-120m" else EsmForMaskedLM
    )
    model = model_cls(config_cls(**kwargs)).to(device).eval()
    inputs = model_family.tokenizer("ACDEFGH", return_tensors="pt").to(device)
    original_forward = model.lm_head.forward.__func__
    head_dtypes = []
    hook = model.lm_head.register_forward_pre_hook(
        lambda module, args: head_dtypes.append(args[0].dtype)
    )
    with torch.no_grad():
        expected = model(**inputs).logits
        actual = masked_lm_logits(model, inputs)
    hook.remove()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-2 if device == "cuda" else 1e-5,
        atol=1e-3 if device == "cuda" else 1e-7,
    )
    assert head_dtypes == [torch.float32, torch.float32]
    assert model.lm_head.forward.__func__ is original_forward
    assert "forward" not in model.lm_head.__dict__


class _ToyMaskedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del attention_mask, position_ids
        vocab = torch.arange(24, device=input_ids.device, dtype=torch.float32)
        next_ids = input_ids.roll(1, dims=1).float()
        logits = (
            input_ids.float().unsqueeze(-1) * vocab
            + next_ids.unsqueeze(-1) * vocab.square()
        )
        return SimpleNamespace(logits=logits + self.weight)


def test_jacobian_reduction_is_symmetric_and_excludes_self_pairs():
    jacobian = torch.randn(17, 20, 17, 20)
    centered = jacobian.clone()
    for axis in range(4):
        centered = centered - centered.mean(dim=axis, keepdim=True)
    symmetric = 0.5 * (centered + centered.permute(2, 3, 0, 1))
    scores = torch.linalg.vector_norm(symmetric, dim=(1, 3))
    scores.fill_diagonal_(0)
    row_sum = scores.sum(dim=-1, keepdim=True)
    col_sum = scores.sum(dim=-2, keepdim=True)
    scores -= row_sum * col_sum / scores.sum()
    scores.fill_diagonal_(0)

    actual = jacobian_to_contacts(jacobian)

    assert actual.shape == (17, 17)
    assert torch.allclose(actual, actual.T, atol=1e-6)
    assert torch.equal(actual.diagonal(), torch.zeros(17))
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, scores, atol=1e-5)


def test_mutant_effects_use_canonical_log_probabilities():
    logits = torch.tensor([[[0.0, 1.0, 2.0]]], dtype=torch.float16)
    residue_ids = torch.tensor([0])
    aa_token_ids = torch.tensor([0, 2])

    actual = _canonical_log_probabilities(logits, residue_ids, aa_token_ids)
    expected = torch.log_softmax(logits.float(), dim=-1)[..., aa_token_ids]

    assert torch.equal(actual, expected)


def test_predict_scores_a_full_contact_chain():
    alignment = align_contact_sequence("ACDE")
    chain = ContactChain(
        record_id="toy",
        split="test",
        source="toy",
        edition=14,
        complex_id=None,
        target=None,
        alignment=alignment,
        units=(),
    )
    tokenizer = _Tokenizer()
    model = ContactJacobianModel(_ToyMaskedLM(), tokenizer, mutant_batch_size=7)

    scores = model.predict(chain, tokenizer)

    assert scores.shape == (4, 4)
    assert torch.allclose(scores, scores.T, atol=1e-6)
    assert torch.isfinite(scores).all()
