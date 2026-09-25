"""Tests for categorical Jacobian zero-shot contact prediction."""

from __future__ import annotations

import pytest
import torch
from transformers import EsmConfig

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.model.categorical_jacobian import (
    CategoricalJacobianConfig,
    CategoricalJacobianModel,
    jacobian_to_contacts,
    resolve_aa_token_ids,
    resolve_special_token_layout,
)
from modules.evaluate.src.tasks import get_task


class _FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    cls_token_id = None
    sep_token_id = None

    def __init__(self) -> None:
        self._mapping = {
            aa: idx
            for idx, aa in enumerate("ACDEFGHIKLMNPQRSTVWY", start=4)
        }

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._mapping[token] for token in tokens]

    def __call__(self, sequence: str, **kwargs: object) -> dict[str, list[int]]:
        del kwargs
        return {"input_ids": [self._mapping[aa] for aa in sequence]}


def _tiny_esm_config() -> EsmConfig:
    return EsmConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=40,
        pad_token_id=1,
        max_position_embeddings=64,
    )


def test_resolve_aa_token_ids_uses_canonical_amino_acids():
    token_ids = resolve_aa_token_ids(_FakeTokenizer())
    assert len(token_ids) == 20
    assert token_ids[0] == 4
    assert token_ids[-1] == 23


def test_resolve_aa_token_ids_rejects_non_residue_tokenization():
    class _MergingTokenizer(_FakeTokenizer):
        def __call__(self, sequence: str, **kwargs: object) -> dict[str, list[int]]:
            del sequence, kwargs
            return {"input_ids": [4, 5]}

    try:
        resolve_aa_token_ids(_MergingTokenizer())
    except ValueError as exc:
        assert "distinct single token" in str(exc)
    else:
        raise AssertionError("Expected residue-tokenization validation to fail.")


def test_resolve_special_token_layout_requires_bos_and_eos():
    tokenizer = _FakeTokenizer()
    tokenizer.eos_token_id = None

    try:
        resolve_special_token_layout(tokenizer)
    except ValueError as exc:
        assert "leading and trailing special token" in str(exc)
    else:
        raise AssertionError("Expected special-token layout validation to fail.")


def test_jacobian_to_contacts_returns_symmetric_map():
    jacobian = torch.randn(6, 20, 6, 20)
    contacts = jacobian_to_contacts(jacobian)
    assert contacts.shape == (6, 6)
    assert torch.allclose(contacts, contacts.transpose(0, 1), atol=1e-5)


def test_jacobian_to_contacts_uses_pre_norm_directional_symmetrization():
    jacobian = torch.randn(4, 20, 4, 20)
    centered = jacobian.float()
    for _ in range(20):
        for axis in range(4):
            centered = centered - centered.mean(dim=axis, keepdim=True)

    symmetric = 0.5 * (centered + centered.permute(2, 3, 0, 1))
    expected = torch.linalg.vector_norm(symmetric, dim=(1, 3))
    off_diag = ~torch.eye(4, dtype=torch.bool)
    masked_scores = expected * off_diag
    row_mean = masked_scores.sum(dim=1, keepdim=True) / 3
    col_mean = masked_scores.sum(dim=0, keepdim=True) / 3
    apc = row_mean * col_mean / (masked_scores.sum() / 12).clamp_min(1e-8)
    expected = (expected - apc) * off_diag

    assert torch.allclose(jacobian_to_contacts(jacobian), expected, atol=1e-5)


def test_jacobian_to_contacts_removes_categorical_main_effects():
    mutation_site = torch.randn(3, 1, 1, 1)
    mutation_residue = torch.randn(1, 20, 1, 1)
    output_site = torch.randn(1, 1, 3, 1)
    output_residue = torch.randn(1, 1, 1, 20)
    jacobian = mutation_site + mutation_residue + output_site + output_residue

    assert torch.allclose(jacobian_to_contacts(jacobian), torch.zeros(3, 3), atol=1e-5)


def test_forward_is_chunking_invariant_and_padding_stable():
    aa_token_ids = tuple(range(4, 24))
    small_chunks = CategoricalJacobianModel(
        CategoricalJacobianConfig(
            trunk_config=_tiny_esm_config(), aa_token_ids=aa_token_ids, mutant_batch_size=3
        )
    )
    large_chunks = CategoricalJacobianModel(
        CategoricalJacobianConfig(
            trunk_config=_tiny_esm_config(),
            aa_token_ids=aa_token_ids,
            mutant_batch_size=1000,
        )
    )
    large_chunks.load_state_dict(small_chunks.state_dict())
    small_chunks.eval()
    large_chunks.eval()

    input_ids = torch.tensor([[0, 4, 5, 6, 7, 2, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)

    logits_small = small_chunks(input_ids=input_ids, attention_mask=attention_mask).logits
    logits_large = large_chunks(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.allclose(logits_small, logits_large, atol=1e-5)

    trimmed_logits = small_chunks(
        input_ids=input_ids[:, :6],
        attention_mask=attention_mask[:, :6],
    ).logits
    assert torch.allclose(logits_small[:, :6, :6], trimmed_logits, atol=1e-5)

    map_min = logits_small.min()
    assert torch.all(logits_small[:, 6:, :] == map_min)
    assert torch.all(logits_small[:, :, 6:] == map_min)


def test_forward_rejects_unexpected_special_token_layout():
    model = CategoricalJacobianModel(
        CategoricalJacobianConfig(
            trunk_config=_tiny_esm_config(),
            aa_token_ids=tuple(range(4, 24)),
            leading_token_id=0,
            trailing_token_id=2,
        )
    )
    input_ids = torch.tensor([[3, 4, 5, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with pytest.raises(ValueError, match="leading/trailing special token layout"):
        model(input_ids=input_ids, attention_mask=attention_mask)


def test_model_rejects_invalid_batch_size_and_oversized_jacobian():
    config = _tiny_esm_config()
    try:
        CategoricalJacobianModel(
            CategoricalJacobianConfig(
                trunk_config=config, aa_token_ids=tuple(range(4, 24)), mutant_batch_size=0
            )
        )
    except ValueError as exc:
        assert "mutant_batch_size" in str(exc)
    else:
        raise AssertionError("Expected invalid mutant batch size to fail.")

    model = CategoricalJacobianModel(
        CategoricalJacobianConfig(
            trunk_config=config,
            aa_token_ids=tuple(range(4, 24)),
            max_jacobian_bytes=1,
        )
    )
    input_ids = torch.tensor([[0, 4, 5, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    except ValueError as exc:
        assert "Jacobian tensor" in str(exc)
    else:
        raise AssertionError("Expected oversized Jacobian to fail.")


def test_task_extract_predictions_is_identity():
    task = get_task("categorical_jacobian")
    logits = torch.randn(2, 4, 4)
    predictions = task.extract_predictions(logits)
    assert torch.equal(predictions, logits)


def test_task_build_model_passes_model_kwargs(monkeypatch):
    task = get_task("categorical_jacobian")
    captured: dict[str, object] = {}

    def fake_builder(model_id: str, **kwargs: object) -> object:
        captured["model_id"] = model_id
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "modules.evaluate.src.tasks.categorical_jacobian.build_categorical_jacobian_model",
        fake_builder,
    )

    model = task.build_model(
        "toy-model",
        num_labels=2,
        model_kwargs={"mutant_batch_size": 8, "max_jacobian_bytes": 1234},
    )

    assert model is not None
    assert captured == {
        "model_id": "toy-model",
        "mutant_batch_size": 8,
        "max_jacobian_bytes": 1234,
    }


def test_categorical_jacobian_requires_evaluate_as_is_mode():
    payload = {
        "workspace": {
            "dataset_name": "toy_dataset",
            "model_name": "toy_model",
            "task_type": "categorical_jacobian",
            "target_type": "categorical",
            "label_format": "contact_pairs",
            "num_labels": None,
        },
        "steps": {"mode": "tune_probe"},
    }

    try:
        EvaluationConfig.model_validate(payload)
    except ValueError as exc:
        assert "categorical_jacobian" in str(exc)
        assert "evaluate_as_is" in str(exc)
    else:
        raise AssertionError("Expected categorical_jacobian mode validation to fail.")


def test_separation_prior_allowed_for_categorical_jacobian():
    payload = {
        "workspace": {
            "dataset_name": "toy_dataset",
            "model_name": "toy_model",
            "task_type": "categorical_jacobian",
            "target_type": "categorical",
            "label_format": "contact_pairs",
            "num_labels": None,
        },
        "steps": {
            "mode": "evaluate_as_is",
            "predict": {"controls": ["separation_prior"]},
        },
    }

    config = EvaluationConfig.model_validate(payload)
    assert config.steps.predict.controls == ["separation_prior"]
