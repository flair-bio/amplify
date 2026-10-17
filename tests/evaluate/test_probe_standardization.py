"""Training-only feature standardization for frozen torch probes."""

import json
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from modules.evaluate.src.model.heads import EmbeddingProbeHead, HeadConfig
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.tune import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
    TuneStep,
)
from modules.evaluate.src.utils.embed import EmbeddingDataset

from .test_prepare import _make_workspace

_FEATURES = torch.tensor(
    [[2.0, 10.0, 7.0], [4.0, 14.0, 7.0], [6.0, 18.0, 7.0], [8.0, 22.0, 7.0]]
)


def _loader(features, labels):
    return DataLoader(
        EmbeddingDataset(features, list(range(len(features))), labels), batch_size=2
    )


@pytest.mark.parametrize("hidden_size", [None, 5])
def test_standardization_conditions_actual_head_inputs(hidden_size):
    model = EmbeddingProbeHead(3, 2, "sequence_classification", hidden_size)
    model.fit_standardization(_FEATURES)
    observed = []
    hook = model.projection.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    try:
        output = model(inputs_embeds=_FEATURES)
    finally:
        hook.remove()

    transformed = observed[0]
    torch.testing.assert_close(transformed.mean(dim=0), torch.zeros(3))
    torch.testing.assert_close(
        transformed.std(dim=0, correction=0), torch.tensor([1.0, 1.0, 0.0])
    )
    assert torch.isfinite(output.logits).all()
    assert model.feature_std[-1] > 0


@pytest.mark.parametrize("hidden_size", [None, 5])
def test_disabled_standardization_preserves_legacy_outputs(hidden_size):
    model = EmbeddingProbeHead(3, 2, "sequence_classification", hidden_size)
    with torch.inference_mode():
        expected = model.classifier(model.projection(_FEATURES))
    model.fit_standardization(_FEATURES)
    model.fit_standardization(_FEATURES, enabled=False)

    with torch.inference_mode():
        actual = model(inputs_embeds=_FEATURES).logits

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.feature_mean is None
    assert model.feature_std is None
    assert "feature_mean" not in model.state_dict()
    assert "feature_std" not in model.state_dict()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("hidden_size", [None, 5])
def test_standardization_state_roundtrip(enabled, hidden_size):
    model = EmbeddingProbeHead(
        3, 2, "sequence_classification", hidden_size, standardize_features=enabled
    )
    model.fit_standardization(_FEATURES, enabled=enabled)
    restored = EmbeddingProbeHead(
        3, 2, "sequence_classification", hidden_size, standardize_features=enabled
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    held_out = _FEATURES + torch.tensor([100.0, -20.0, 0.0])

    with torch.inference_mode():
        torch.testing.assert_close(
            restored(inputs_embeds=held_out).logits,
            model(inputs_embeds=held_out).logits,
        )


def test_feature_standardization_keeps_regression_targets_in_original_units():
    model = EmbeddingProbeHead(3, 1, "sequence_regression")
    model.fit_standardization(_FEATURES)
    targets = torch.tensor([10.0, 20.0, 30.0, 40.0])

    output = model(inputs_embeds=_FEATURES, labels=targets)

    torch.testing.assert_close(
        output.loss, torch.nn.functional.mse_loss(output.logits.squeeze(-1), targets)
    )


@pytest.mark.parametrize("head_type", ["mlp", "torch_linear"])
@pytest.mark.parametrize("enabled", [False, True])
def test_tuning_standardizes_training_features_and_persists_policy(
    tmp_path, head_type, enabled
):
    features = _FEATURES.clone()
    labels = torch.tensor([0, 1, 0, 1])
    prepared = PreparedArtifacts(
        model=EmbeddingProbeHead(3, 2, "sequence_classification"),
        tokenizer=MagicMock(),
        train_dataloader=_loader(features, labels),
        val_dataloader=_loader(features + 1000, labels),
        test_dataloader=_loader(features + 2000, labels),
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    options = {} if enabled else {"standardize_features": False}
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type=head_type, hidden_size=5),
            max_epochs=1,
            early_stopping_patience=None,
            **options,
        )
    )

    result = step.run(workspace, prepared, accelerator=Accelerator(cpu=True))

    if enabled:
        torch.testing.assert_close(
            result.model.feature_mean, torch.tensor([5.0, 16.0, 7.0])
        )
        torch.testing.assert_close(
            result.model.feature_std[:2],
            torch.tensor([5.0, 20.0]).sqrt(),
        )
    else:
        assert result.model.feature_mean is None
        assert result.model.feature_std is None
    torch.testing.assert_close(features, _FEATURES, rtol=0, atol=0)
    metadata = json.loads((workspace.model_path / "probe_head_config.json").read_text())
    assert metadata["standardize_features"] is enabled
    restored = EmbeddingProbeHead(
        metadata["embedding_size"],
        metadata["num_labels"],
        metadata["task_type"],
        probe_hidden_size=metadata["probe_hidden_size"],
        standardize_features=metadata["standardize_features"],
    )
    restored.load_state_dict(
        torch.load(workspace.model_path / "probe_head.pt", weights_only=True),
        strict=True,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(inputs_embeds=features + 1000).logits,
            result.model(inputs_embeds=features + 1000).logits,
        )


def test_token_standardization_ignores_unlabeled_positions(tmp_path):
    features = torch.cat([torch.full((1, 3), 1e9), _FEATURES])
    labels = torch.tensor([-100, 0, 1, 0, 1])
    prepared = PreparedArtifacts(
        model=EmbeddingProbeHead(3, 2, "token_classification"),
        tokenizer=MagicMock(),
        train_dataloader=_loader(features, labels),
        val_dataloader=None,
        test_dataloader=None,
    )
    workspace = _make_workspace(tmp_path, "token_classification")
    step = TuneStep(
        TuneConfig(max_epochs=1, early_stopping_patience=None, save_best_model=False)
    )

    trained = step.run(workspace, prepared, accelerator=Accelerator(cpu=True))

    torch.testing.assert_close(
        trained.model.feature_mean, torch.tensor([5.0, 16.0, 7.0])
    )
    torch.testing.assert_close(
        trained.model.feature_std[:2], torch.tensor([5.0, 20.0]).sqrt()
    )


def test_hpo_and_seed_variance_keep_training_standardization(monkeypatch, tmp_path):
    labels = torch.tensor([0, 1, 0, 1])
    prepared = PreparedArtifacts(
        model=EmbeddingProbeHead(3, 2, "sequence_classification"),
        tokenizer=MagicMock(),
        train_dataloader=_loader(_FEATURES.clone(), labels),
        val_dataloader=_loader(_FEATURES + 1000, labels),
        test_dataloader=None,
    )
    fits = []
    fit_standardization = EmbeddingProbeHead.fit_standardization

    def record_fit(self, features, **kwargs):
        fits.append(features.clone())
        return fit_standardization(self, features, **kwargs)

    monkeypatch.setattr(EmbeddingProbeHead, "fit_standardization", record_fit)
    step = TuneStep(
        TuneConfig(
            max_epochs=1,
            early_stopping_patience=None,
            save_best_model=False,
            variance_seeds=[41],
            hyperparameter_search=HyperparameterSearchConfig(
                enabled=True,
                n_trials=2,
                search_pattern="grid",
                search_space=HyperparameterSearchSpaceConfig(
                    learning_rate=[1e-3, 1e-2], max_epochs=[1], weight_decay=[0.0]
                ),
            ),
        )
    )

    trained = step.run(
        _make_workspace(tmp_path, "sequence_classification"),
        prepared,
        accelerator=Accelerator(cpu=True),
    )

    assert len(fits) == 1
    torch.testing.assert_close(fits[0], _FEATURES)
    torch.testing.assert_close(
        trained.model.feature_mean, torch.tensor([5.0, 16.0, 7.0])
    )
    torch.testing.assert_close(
        trained.model.feature_std[:2], torch.tensor([5.0, 20.0]).sqrt()
    )
