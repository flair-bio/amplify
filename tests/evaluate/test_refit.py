"""Final cached probes refit from scratch on development data, never test data."""

import json
import math

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config.evaluation import (
    EvaluationConfig,
    HeadConfig,
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
)
from modules.evaluate.src.dataset.embeddings import EmbeddingDataset
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.trainers.fitting import _report_torch_seed_variance
from modules.evaluate.src.trainers.torch import train_torch

from .helpers import fit_inputs, fitting_state, make_inputs, patch_function
from .test_prepare import _make_workspace

TRAIN = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
VALIDATION = torch.tensor([[10.0, 12.0], [14.0, 16.0]])
TEST = torch.full((2, 2), 10000.0)


def prepared_splits(task_type="sequence_classification"):

    def loader(features, labels, prefix):
        return DataLoader(
            EmbeddingDataset(
                features.clone(),
                [f"{prefix}-{i}" for i in range(len(features))],
                labels,
            ),
            batch_size=2,
        )

    return make_inputs(
        model=EmbeddingProbeHead(2, 2, task_type),
        tokenizer=None,
        train_dataloader=loader(TRAIN, [0, 1, 0], "train"),
        val_dataloader=loader(VALIDATION, [1, 0], "val"),
        test_dataloader=loader(TEST, [1, 1], "test"),
    )


@pytest.mark.parametrize("head_type", ["mlp", "torch_linear"])
@pytest.mark.parametrize("search_enabled", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_fresh_refit_uses_selected_architecture_budget_and_development_statistics(
    monkeypatch, tmp_path, head_type, search_enabled, wrapped
):
    prepared = prepared_splits()
    accelerator = Accelerator(cpu=True)
    if wrapped:
        (
            prepared.training.train_dataloader,
            prepared.training.val_dataloader,
            prepared.test.test_dataloader,
        ) = accelerator.prepare(
            prepared.training.train_dataloader,
            prepared.training.val_dataloader,
            prepared.test.test_dataloader,
        )
    original_loaders = (
        prepared.training.train_dataloader,
        prepared.training.val_dataloader,
        prepared.test.test_dataloader,
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    config = TuneConfig(
        head=HeadConfig(head_type=head_type, hidden_size=3),
        refit_on_train_validation=True,
        standardize_features=True,
        max_epochs=2,
        batch_size=2,
        early_stopping_patience=1,
        variance_seeds=[123],
        hyperparameter_search=HyperparameterSearchConfig(
            enabled=search_enabled,
            search_pattern="grid",
            search_space=HyperparameterSearchSpaceConfig(
                hidden_size=[3, 5] if head_type == "mlp" else None,
                hidden_depth=[1, 2] if head_type == "mlp" else None,
                batch_size=[2, 4] if head_type == "mlp" else None,
                learning_rate=[0.001],
                weight_decay=[0.0],
                max_epochs=[1, 2],
            ),
        ),
    )
    trials, refits, schedules = ([], [], [])

    def record_train(
        config, train_trunk, ws, predictor, training, accelerator, hparams, **kwargs
    ):
        model = predictor.model
        if training.val_dataloader is not None:
            assert training.train_dataloader is original_loaders[0]
            torch.testing.assert_close(model.feature_mean, TRAIN.mean(dim=0))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(0.25)
            trials.append(model)
            return float(
                hparams.get("hidden_size", 1)
                + hparams.get("hidden_depth", 0)
                + hparams.get("batch_size", 0)
                + hparams["max_epochs"]
            )
        assert config.early_stopping_patience is None
        assert model not in trials
        dataset = training.train_dataloader.dataset
        torch.testing.assert_close(dataset.embeddings, torch.cat([TRAIN, VALIDATION]))
        assert dataset.ids == ["train-0", "train-1", "train-2", "val-0", "val-1"]
        assert dataset.labels.tolist() == [0, 1, 0, 1, 0]
        torch.testing.assert_close(model.feature_mean, dataset.embeddings.mean(dim=0))
        torch.testing.assert_close(
            model.feature_std, dataset.embeddings.std(dim=0, correction=0)
        )
        assert not torch.all(model.classifier.weight == 0.25)
        refits.append((model, dict(hparams)))
        return train_torch(
            config, train_trunk, ws, predictor, training, accelerator, hparams, **kwargs
        )

    def scheduler(*args, **kwargs):
        schedules.append(kwargs["num_training_steps"])
        return get_scheduler(*args, **kwargs)

    def variance(
        config, train_trunk, _fit_result, ws, predictor, training, accelerator, hparams
    ):
        assert training.train_dataloader is original_loaders[0]
        assert training.val_dataloader is original_loaders[1]
        torch.testing.assert_close(predictor.model.feature_mean, TRAIN.mean(dim=0))

    patch_function(monkeypatch, train_torch, record_train)
    patch_function(monkeypatch, _report_torch_seed_variance, variance)
    monkeypatch.setattr("modules.evaluate.src.trainers.torch.get_scheduler", scheduler)
    result = fit_inputs(config, workspace, prepared, seed=42, accelerator=accelerator)
    assert len(refits) == 1
    model, hparams = refits[0]
    assert result.predictor.model is model
    if head_type == "mlp" and search_enabled:
        assert model.probe_hidden_size == 5
        assert model.probe_hidden_depth == 2
        assert hparams["batch_size"] == 4
    assert hparams["max_epochs"] == 2
    assert schedules == [math.ceil(5 / hparams.get("batch_size", 2)) * 2]
    assert (
        result.training.train_dataloader,
        result.training.val_dataloader,
        result.test.test_dataloader,
    ) == original_loaders
    assert config.early_stopping_patience == 1
    metadata = json.loads((workspace.model_path / "probe_head_config.json").read_text())
    assert metadata["final_fit_split"] == "train+validation"
    assert metadata["best_hyperparameters"] == hparams
    restored = EmbeddingProbeHead(
        metadata["embedding_size"],
        metadata["num_labels"],
        metadata["task_type"],
        probe_hidden_size=metadata["probe_hidden_size"],
        probe_hidden_depth=metadata["probe_hidden_depth"],
        standardize_features=metadata["standardize_features"],
    )
    restored.load_state_dict(
        torch.load(workspace.model_path / "probe_head.pt", weights_only=True)
    )
    torch.testing.assert_close(
        restored(inputs_embeds=TEST).logits, model(inputs_embeds=TEST).logits
    )


def test_classical_refit_rebuilds_estimator_and_scaler_on_development_data(
    monkeypatch, tmp_path
):
    from modules.evaluate.src.model.classical import build_estimator

    estimators = []

    def build(*args, **kwargs):
        estimator = build_estimator(*args, **kwargs)
        estimators.append(estimator)
        return estimator

    monkeypatch.setattr(
        "modules.evaluate.src.trainers.classical.build_estimator", build
    )
    prepared = prepared_splits()
    workspace = _make_workspace(tmp_path, "sequence_classification")
    config = TuneConfig(
        head=HeadConfig(head_type="sklearn_linear", search_space={"C": [0.1, 1.0]}),
        refit_on_train_validation=True,
        metric_aggregation="macro",
    )
    result = fit_inputs(
        config, workspace, prepared, seed=42, accelerator=Accelerator(cpu=True)
    )
    assert len(estimators) == 3
    assert result.predictor.model is estimators[-1]
    for estimator in estimators[:-1]:
        np.testing.assert_allclose(estimator[0].mean_, TRAIN.mean(dim=0))
    np.testing.assert_allclose(
        result.predictor.model[0].mean_, torch.cat([TRAIN, VALIDATION]).mean(dim=0)
    )
    assert result.predictor.model[0].n_samples_seen_ == 5
    metadata = json.loads((workspace.model_path / "tune_config.json").read_text())
    assert metadata["final_fit_split"] == "train+validation"
    assert result.predictor.model[-1].C == metadata["best_hyperparameters"]["C"]


def test_final_refit_includes_additional_split_after_hyperparameter_selection(
    monkeypatch, tmp_path
):
    prepared = prepared_splits()
    excluded = DataLoader(
        EmbeddingDataset(torch.tensor([[20.0, 22.0]]), ["tuning-excluded"], [1]),
        batch_size=2,
    )
    prepared.training.refit_dataloaders = {"tuning_excluded": excluded}
    prepared.training.refit_token_dataloaders = {"tuning_excluded": excluded}
    selection_sizes = []
    refit_ids = []

    def record_train(
        config,
        train_trunk,
        workspace,
        predictor,
        training,
        accelerator,
        hparams,
        **kwargs,
    ):
        train = training.train_dataloader.dataset
        if training.val_dataloader is not None:
            selection_sizes.append(len(train))
            assert "tuning-excluded" not in train.ids
            return 0.5
        refit_ids.extend(train.ids)
        return train_torch(
            config,
            train_trunk,
            workspace,
            predictor,
            training,
            accelerator,
            hparams,
            **kwargs,
        )

    patch_function(monkeypatch, train_torch, record_train)
    config = TuneConfig(
        head=HeadConfig(head_type="mlp", hidden_size=3),
        refit_on_train_validation=True,
        max_epochs=1,
        batch_size=2,
        save_best_model=False,
        hyperparameter_search=HyperparameterSearchConfig(
            enabled=True,
            search_pattern="grid",
            search_space=HyperparameterSearchSpaceConfig(
                learning_rate=[0.001], weight_decay=[0.0], max_epochs=[1]
            ),
        ),
    )
    fit_inputs(
        config,
        _make_workspace(tmp_path, "sequence_classification"),
        prepared,
        seed=42,
        accelerator=Accelerator(cpu=True),
        publish=False,
        state=fitting_state(config),
    )
    assert selection_sizes == [len(TRAIN)]
    assert refit_ids == [
        "train-0",
        "train-1",
        "train-2",
        "val-0",
        "val-1",
        "tuning-excluded",
    ]


@pytest.mark.parametrize(
    "task_type",
    [
        "token_classification",
        "sequence_multilabel_classification",
        "sequence_regression",
    ],
)
def test_refit_preserves_target_shapes_masks_and_seed_reproducibility(
    tmp_path, task_type
):
    states = []
    for _ in range(2):
        prepared = prepared_splits(task_type)
        num_labels = 1 if task_type == "sequence_regression" else 2
        prepared.predictor.model = EmbeddingProbeHead(2, num_labels, task_type)
        if task_type == "sequence_multilabel_classification":
            prepared.training.train_dataloader.dataset.labels = [[1, 0], [0, 1], [1, 1]]
            prepared.training.val_dataloader.dataset.labels = [[1, 1], [0, 0]]
        elif task_type == "token_classification":
            prepared.training.val_dataloader.dataset.labels = [-100, 0]
        step = fitting_state(
            TuneConfig(
                refit_on_train_validation=True,
                max_epochs=2,
                batch_size=128,
                save_best_model=False,
                head=HeadConfig(hidden_size=3, hidden_depth=2),
            )
        )
        result = fit_inputs(
            step.config,
            _make_workspace(tmp_path, task_type),
            prepared,
            seed=42,
            accelerator=Accelerator(cpu=True),
            mode="finetune" if step.train_trunk else "tune_probe",
            publish=False,
            state=step,
        )
        features = torch.cat([TRAIN, VALIDATION])
        if task_type == "token_classification":
            features = features[[0, 1, 2, 4]]
        torch.testing.assert_close(
            result.predictor.model.feature_mean, features.mean(dim=0)
        )
        assert torch.isfinite(result.predictor.model(inputs_embeds=TEST).logits).all()
        states.append(
            {
                key: value.clone()
                for key, value in result.predictor.model.state_dict().items()
            }
        )
    for name, value in states[0].items():
        torch.testing.assert_close(states[1][name], value, rtol=0, atol=0)


@pytest.mark.parametrize("validation", [None, "empty"])
def test_refit_rejects_missing_or_empty_validation_before_training(
    tmp_path, validation
):
    prepared = prepared_splits()
    prepared.training.val_dataloader = (
        None
        if validation is None
        else DataLoader(EmbeddingDataset(torch.empty(0, 2), [], []))
    )
    with pytest.raises(ValueError, match="requires nonempty train and validation"):
        fit_inputs(
            TuneConfig(refit_on_train_validation=True),
            _make_workspace(tmp_path, "sequence_classification"),
            prepared,
            accelerator=Accelerator(cpu=True),
            publish=False,
            state=fitting_state(TuneConfig(refit_on_train_validation=True)),
        )


@pytest.mark.parametrize("unsupported", ["live", "layerwise", "finetune"])
def test_refit_config_rejects_unsupported_paths(unsupported):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    overrides = ["steps.tune.hyperparameter_search.enabled=false"]
    if unsupported == "layerwise":
        overrides += [
            "steps.layerwise.enabled=true",
            "steps.tune.head.head_type=torch_linear",
        ]
    else:
        overrides += ["steps.prepare.embedding_cache.enabled=false"]
        if unsupported == "finetune":
            overrides += ["steps.mode=finetune"]
    with pytest.raises(ValueError, match="refit_on_train_validation requires"):
        load_and_parse(
            [
                root / "config.yaml",
                root / "tasks/sequence_classification.yaml",
                root / "datasets/fold_prediction.yaml",
            ],
            EvaluationConfig,
            overrides=overrides,
        )
