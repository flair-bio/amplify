import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from transformers import get_scheduler

from modules.evaluate.src.artifacts.cache import EmbeddingCache
from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config import (
    HeadConfig,
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    LayerwiseConfig,
    TuneConfig,
)
from modules.evaluate.src.config import EmbeddingCacheConfig
from modules.evaluate.src.dataset.embeddings import EmbeddingDataset
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.trainers.classical_estimators import _gather_embeddings, fit_candidates
from modules.evaluate.src.trainers.fitting import _report_torch_seed_variance
from modules.evaluate.src.trainers.layerwise import fit_layerwise, improves
from modules.evaluate.src.trainers.selection import (
    _generate_candidates,
    _search_hyperparameters,
    _suggest_weight_decay,
)
from modules.evaluate.src.trainers.selection_helpers import (
    build_optimizer,
    checkpoint_state,
    resolve_search_direction,
    sample_candidates,
    split_decay_parameters,
)
from modules.evaluate.src.trainers.torch import (
    _cached_embedding_split,
    evaluate_torch_validation,
    resolve_selection_metric,
    train_torch,
)

from .helpers import fit_inputs, fitting_state, make_inputs, patch_function
from .test_prepare import _make_workspace, _run_prepare


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "task_type,num_labels",
    [
        ("sequence_classification", 2),
        ("token_classification", 3),
        ("token_classification", 8),
        ("sequence_multilabel_classification", 3),
    ],
)
def test_validation_f1_keeps_configured_classes_and_token_masks(
    monkeypatch, tmp_path, cached, task_type, num_labels
):
    labels = torch.zeros(3, dtype=torch.long)
    if task_type == "token_classification":
        labels[1] = -100
    elif task_type == "sequence_multilabel_classification":
        labels = torch.tensor([[1, 0, 0]] * 3)
    model = EmbeddingProbeHead(4, num_labels, task_type)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-5)
        model.classifier.bias[0] = 5
    loader = DataLoader(
        EmbeddingDataset(torch.ones(3, 4), list(range(3)), labels)
        if cached
        else [{"input_ids": torch.tensor([0]), "labels": 0}],
        batch_size=2,
    )
    prepared = make_inputs(
        model=model,
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=loader,
        test_dataloader=None,
    )
    live_labels = (
        [[0, 0], []] if task_type == "token_classification" else labels.tolist()
    )
    monkeypatch.setattr(
        "modules.evaluate.src.trainers.torch.run_inference",
        lambda *args, **kwargs: ([], live_labels, live_labels, None, None),
    )
    step = fitting_state(TuneConfig(metric_aggregation="macro"))
    score = evaluate_torch_validation(
        step.config,
        _make_workspace(tmp_path, task_type, num_labels=num_labels),
        prepared.predictor,
        prepared.training,
        Accelerator(cpu=True),
        "f1",
    )
    assert score == pytest.approx(1 / num_labels)


@pytest.mark.parametrize("cached", [False, True])
def test_validation_fmax_collects_sigmoid_probabilities(monkeypatch, tmp_path, cached):
    task_type = "sequence_multilabel_classification"
    probabilities = torch.tensor([[0.4, 0.1, 0.1], [0.1, 0.4, 0.4]])
    labels = torch.tensor([[1, 0, 0], [0, 1, 1]])
    model = EmbeddingProbeHead(3, 3, task_type)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(3))
        model.classifier.bias.zero_()
    loader = DataLoader(
        EmbeddingDataset(torch.logit(probabilities), [0, 1], labels)
        if cached
        else [{"input_ids": torch.tensor([0]), "labels": 0}],
        batch_size=1,
    )
    prepared = make_inputs(
        model=model,
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=loader,
        test_dataloader=None,
    )

    def run_inference(*args, **kwargs):
        assert kwargs["collect_probabilities"]
        return (
            [],
            torch.zeros_like(labels).tolist(),
            labels.tolist(),
            probabilities.tolist(),
            None,
        )

    monkeypatch.setattr(
        "modules.evaluate.src.trainers.torch.run_inference", run_inference
    )
    step = fitting_state(TuneConfig(metric_aggregation="macro"))
    score = evaluate_torch_validation(
        step.config,
        _make_workspace(tmp_path, task_type, num_labels=3),
        prepared.predictor,
        prepared.training,
        Accelerator(cpu=True),
        "fmax",
    )
    assert score == pytest.approx(1.0)


def test_validation_hpo_uses_grouped_regression_metric(monkeypatch, tmp_path):
    ids = ["a1", "a2", "b1", "b2"]
    predictions = [0.0, 1.0, 2.0, 3.0]
    labels = [0.0, 1.0, 3.0, 2.0]
    loader = [{"input_ids": torch.tensor([0]), "labels": 0.0}]
    prepared = make_inputs(
        model=nn.Linear(1, 1),
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=loader,
        test_dataloader=None,
        validation_metric_groups={
            json.dumps(identifier): group
            for identifier, group in zip(ids, ["a", "a", "b", "b"], strict=True)
        },
    )
    monkeypatch.setattr(
        "modules.evaluate.src.trainers.torch.run_inference",
        lambda *args, **kwargs: (ids, predictions, labels, None, None),
    )
    step = fitting_state(TuneConfig(metric_aggregation="macro"))
    score = evaluate_torch_validation(
        step.config,
        _make_workspace(tmp_path, "sequence_regression"),
        prepared.predictor,
        prepared.training,
        Accelerator(cpu=True),
        "spearmanr",
    )
    assert score == pytest.approx(0.0)


def test_classical_hpo_uses_grouped_regression_metric(monkeypatch):

    class Estimator:
        def fit(self, features, labels):
            return self

        def predict(self, features):
            return features[:, 0]

    monkeypatch.setattr(
        "modules.evaluate.src.trainers.classical_estimators.build_estimator",
        lambda *args, **kwargs: Estimator(),
    )
    _, _, score = fit_candidates(
        head_type="sklearn_linear",
        task_type="sequence_regression",
        train_X=np.zeros((2, 1)),
        train_y=np.zeros(2),
        val_X=np.array([[0.0], [1.0], [2.0], [3.0]]),
        val_y=np.array([0.0, 1.0, 3.0, 2.0]),
        search_space={},
        default_hyperparameters={},
        metric_name="spearmanr",
        direction="maximize",
        group_ids=["a", "a", "b", "b"],
    )
    assert score == pytest.approx(0.0)


def test_classical_selection_uses_fixed_class_macro_f1(monkeypatch):
    from modules.evaluate.src.trainers.classical_estimators import (
        fit_candidates as sweep_classical_head,
    )

    candidates = {
        "observed_winner": np.zeros(8, dtype=int),
        "fixed_winner": np.array([0, 2, 2, 2, 1, 2, 2, 2]),
    }

    def build(head_type, task_type, hparams, **kwargs):
        return SimpleNamespace(
            fit=lambda *args: None,
            predict=lambda *args: candidates[hparams["candidate"]],
        )

    monkeypatch.setattr(
        "modules.evaluate.src.trainers.classical_estimators.build_estimator", build
    )
    _, best, score = sweep_classical_head(
        "knn",
        "sequence_classification",
        np.ones((8, 2)),
        np.array([0] * 4 + [1] * 4),
        np.ones((8, 2)),
        np.array([0] * 4 + [1] * 4),
        {"candidate": list(candidates)},
        {},
        "f1",
        "maximize",
        search_pattern="grid",
        average="macro",
        num_labels=3,
    )
    assert best["candidate"] == "fixed_winner"
    assert score == pytest.approx(0.8 / 3)


def test_online_layerwise_tuning_saves_independent_heads(
    monkeypatch, tmp_path, model_family
):

    def cache_access_forbidden(*args, **kwargs):
        pytest.fail("Online layerwise probing must not access the embedding cache.")

    monkeypatch.setattr(EmbeddingCache, "read", cache_access_forbidden)
    monkeypatch.setattr(EmbeddingCache, "write", cache_access_forbidden)
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        layerwise_config=LayerwiseConfig(
            enabled=True, layers=[0, -1], autocast_dtype="float32"
        ),
    )
    assert isinstance(prepared.predictor.model, LayerwiseProbeModel)
    workspace_config = _make_workspace(tmp_path, "sequence_classification").config
    workspace = EvaluationWorkspace(workspace_config, variant="layerwise-test")
    step = fitting_state(
        TuneConfig(
            head=HeadConfig(head_type="torch_linear"),
            max_epochs=1,
            early_stopping_patience=None,
        )
    )
    fit_calls = []

    def record_fit(config, *args, **kwargs):
        fit_calls.append(args[0])
        return fit_layerwise(config, *args, **kwargs)

    patch_function(monkeypatch, fit_layerwise, record_fit)
    trained = fit_inputs(
        step.config,
        workspace,
        prepared,
        accelerator=Accelerator(cpu=True),
        mode="finetune" if step.train_trunk else "tune_probe",
    )
    assert fit_calls == [workspace.config]
    assert isinstance(trained.predictor.model, LayerwiseProbeModel)
    assert all(
        (
            parameter.grad is None
            for parameter in trained.predictor.model.trunk.parameters()
        )
    )
    for layer in (0, 2):
        layer_workspace = EvaluationWorkspace(
            workspace_config, variant=f"layerwise-test-layer_{layer}"
        )
        checkpoint = torch.load(
            layer_workspace.model_path / "probe_head.pt", weights_only=True
        )
        assert checkpoint["layer"] == layer
        assert checkpoint["layerwise_config"]["layers"] == [0, -1]
        assert checkpoint["selection"]["epoch"] == 1
        assert checkpoint["feature_std"].shape == (
            trained.predictor.model.embedding_size,
        )


@pytest.mark.parametrize(
    "score, best, direction, expected",
    [
        (float("nan"), None, "maximize", False),
        (0.1, None, "maximize", True),
        (0.1, float("nan"), "maximize", True),
        (0.1, 0.2, "maximize", False),
        (0.1, 0.2, "minimize", True),
    ],
)
def test_layerwise_improvement_ignores_nan(score, best, direction, expected):
    assert improves(score, best, direction) is expected


def test_checkpoint_state_is_independent_cpu_copy() -> None:
    model = nn.Linear(2, 1)

    checkpoint = checkpoint_state(model)

    assert all(tensor.device.type == "cpu" for tensor in checkpoint.values())
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(checkpoint["weight"], model.weight.cpu())


@pytest.mark.parametrize("norm_type", ["RMSNorm", "ZeroCenteredRMSNorm"])
@pytest.mark.parametrize("load_checkpoint", [False, True])
def test_finetuning_norm_weight_decay(norm_type, load_checkpoint, tmp_path):
    from transformers import AutoModelForSequenceClassification

    from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
    from modules.pretrain.src.model.modeling_amplify import (
        AMPLIFYForSequenceClassification,
        register_amplify_auto_classes,
    )

    config = AMPLIFYConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        norm_type=norm_type,
        qk_norm=True,
    )
    model = AMPLIFYForSequenceClassification(config)
    if load_checkpoint:
        config.auto_map = register_amplify_auto_classes()
        model.save_pretrained(tmp_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            tmp_path, trust_remote_code=True, local_files_only=True
        )
    frozen = model.amplify.layers[0].input_layernorm.weight
    frozen.requires_grad_(False)
    groups = split_decay_parameters(model, 0.2)
    decay_by_param = {
        id(param): group["weight_decay"]
        for group in groups
        for param in group["params"]
    }
    assert len(decay_by_param) == sum(len(group["params"]) for group in groups)
    assert set(decay_by_param) == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(frozen) not in decay_by_param
    assert decay_by_param[id(model.classifier.weight)] == 0.2
    assert decay_by_param[id(model.classifier.bias)] == 0.0
    assert decay_by_param[id(model.amplify.embed_tokens.weight)] == 0.2
    norms = [module for module in model.modules() if type(module).__name__ == norm_type]
    assert len(norms) == 9  # Block pre-norms, QK norms and final encoder norm.
    for norm in norms:
        if norm.weight.requires_grad:
            expected_decay = 0.2 if norm_type == "ZeroCenteredRMSNorm" else 0.0
            assert decay_by_param[id(norm.weight)] == expected_decay
        if getattr(norm, "bias", None) is not None:
            assert decay_by_param[id(norm.bias)] == 0.0
        if norm_type == "ZeroCenteredRMSNorm":
            with torch.no_grad():
                norm.weight.fill_(0.5)
    optimizer = torch.optim.AdamW(groups, lr=0.1)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for norm in norms:
        if norm_type == "ZeroCenteredRMSNorm":
            expected = 0.49 if norm.weight.requires_grad else 0.5
        else:
            expected = 1.0
        torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, expected))


def test_random_candidates_are_unique_and_bounded_by_grid() -> None:
    search_space = HyperparameterSearchSpaceConfig(
        learning_rate=[0.0001, 0.001], weight_decay=[0.0], max_epochs=[1, 2]
    )
    step = fitting_state(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(search_space=search_space)
        )
    )
    candidates = _generate_candidates(step.config, "random", n_trials=10, seed=42)
    assert len(candidates) == 4
    assert len({tuple(candidate.items()) for candidate in candidates}) == 4


@pytest.mark.parametrize(
    "bounds", [[], [0.0, 0.1], [-1e-6, 0.1], [float("nan")], [float("inf")]]
)
def test_log_weight_decay_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="finite, positive bounds"):
        HyperparameterSearchSpaceConfig(
            weight_decay=bounds, weight_decay_sampling="log"
        )


@pytest.mark.parametrize("cached", [True, False])
def test_tpe_samples_weight_decay_continuously_and_reproducibly(
    monkeypatch, tmp_path, cached
):
    pytest.importorskip("optuna")
    accelerator = Accelerator(cpu=True)
    workspace = _make_workspace(tmp_path, "sequence_classification")
    loader = DataLoader(
        EmbeddingDataset(torch.ones(4, 3), list(range(4)), torch.tensor([0, 1, 0, 1]))
        if cached
        else [{"labels": 0}],
        batch_size=2,
    )
    step = fitting_state(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(
                enabled=True,
                n_trials=4,
                search_space=HyperparameterSearchSpaceConfig(
                    learning_rate=[0.0001],
                    weight_decay=[1e-06, 0.1],
                    weight_decay_sampling="log",
                    max_epochs=[1],
                ),
            )
        )
    )
    samples = []

    def train(
        config,
        train_trunk,
        workspace,
        predictor,
        training,
        accelerator,
        hparams,
        **kwargs,
    ):
        distribution = kwargs["trial"].distributions["weight_decay"]
        assert distribution.log
        assert distribution.low == 1e-06
        assert distribution.high == 0.1
        samples.append(hparams["weight_decay"])
        return hparams["weight_decay"]

    patch_function(monkeypatch, train_torch, train)
    for _ in range(2):
        prepared = make_inputs(
            model=EmbeddingProbeHead(3, 2, "sequence_classification"),
            tokenizer=MagicMock(),
            train_dataloader=loader,
            val_dataloader=loader,
            test_dataloader=None,
        )
        best = _search_hyperparameters(
            step.config,
            step.train_trunk,
            step.result,
            workspace,
            prepared.predictor,
            prepared.training,
            accelerator,
            seed=42,
        )
        assert best["weight_decay"] == max(samples[-4:])
    assert samples[:4] == samples[4:]
    assert all((1e-06 < value < 0.1 for value in samples))
    assert len(set(samples[:4])) > 1


def test_categorical_weight_decay_preserves_zero_and_grid_candidates():
    trial = MagicMock()
    trial.suggest_categorical.return_value = 0.0
    step = fitting_state(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(
                search_space=HyperparameterSearchSpaceConfig(
                    learning_rate=[0.0001], weight_decay=[0.0, 0.001], max_epochs=[1]
                )
            )
        )
    )
    assert _suggest_weight_decay(step.config, trial) == 0.0
    trial.suggest_categorical.assert_called_once_with("weight_decay", [0.0, 0.001])
    trial.suggest_float.assert_not_called()
    for pattern in ("random", "grid"):
        candidates = _generate_candidates(step.config, pattern, n_trials=10, seed=42)
        assert {candidate["weight_decay"] for candidate in candidates} == {0.0, 0.001}
    space = step.config.hyperparameter_search.search_space
    space.weight_decay = [1e-06, 0.1]
    space.weight_decay_sampling = "log"
    for pattern in ("random", "grid"):
        candidates = _generate_candidates(step.config, pattern, n_trials=10, seed=42)
        assert {candidate["weight_decay"] for candidate in candidates} == {1e-06, 0.1}


def test_removed_tuning_aliases_are_rejected() -> None:
    with pytest.raises(Exception, match="num_epochs"):
        TuneConfig(num_epochs=1)
    with pytest.raises(Exception, match="seeds"):
        HyperparameterSearchSpaceConfig(seeds=[42])


def test_sample_candidates_is_deterministic_and_without_replacement() -> None:
    candidates = [{"value": value} for value in range(5)]

    first = sample_candidates(candidates, n_trials=3, seed=42)
    second = sample_candidates(candidates, n_trials=3, seed=42)

    assert first == second
    assert len(first) == len({candidate["value"] for candidate in first}) == 3


def test_gather_embeddings_unwraps_accelerate_dataset_wrappers() -> None:
    embeddings = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 0])
    embedding_dataset = EmbeddingDataset(embeddings, ["a", "b", "c"], labels)

    class DatasetWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

    class LoaderWrapper:
        def __init__(self, dataset):
            self.dataset = dataset

    wrapped_loader = LoaderWrapper(DatasetWrapper(embedding_dataset))
    accelerator = MagicMock()
    actual_embeddings, actual_labels = _gather_embeddings(wrapped_loader, accelerator)
    np.testing.assert_array_equal(actual_embeddings, embeddings.numpy())
    np.testing.assert_array_equal(actual_labels, labels.numpy())
    accelerator.gather_for_metrics.assert_not_called()


@pytest.mark.parametrize(
    "metric,direction,expected",
    [
        ("mse", None, "minimize"),
        ("f1", None, "maximize"),
        ("mse", "maximize", "maximize"),
    ],
)
def test_resolve_search_direction_uses_metric_default_unless_explicit(
    metric, direction, expected
) -> None:
    assert resolve_search_direction(metric, direction) == expected


@pytest.mark.parametrize("is_main_process", [False, True])
def test_tune_stage_publishes_pretrained_model_and_settings(
    monkeypatch, tmp_path, is_main_process
) -> None:
    from modules.evaluate.src.trainers import fitting as selection_module

    class PretrainedLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.save_pretrained = MagicMock()

    model = PretrainedLike()
    tokenizer = MagicMock()
    prepared = make_inputs(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=object(),
        val_dataloader=None,
        test_dataloader=None,
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    config = TuneConfig()
    best_hparams = {"learning_rate": 0.0001, "weight_decay": 0.0001, "max_epochs": 10}
    accelerator = MagicMock(is_main_process=is_main_process)
    accelerator.unwrap_model.return_value = model
    accelerator.get_state_dict.return_value = model.state_dict()
    monkeypatch.setattr(
        selection_module,
        "fit_predictor",
        lambda **kwargs: FitResult(
            model=model, seed=710019, selected_hyperparameters=best_hparams
        ),
    )
    fit_inputs(config, workspace, prepared, accelerator=accelerator)
    assert workspace.model_path.is_dir()
    model.save_pretrained.assert_called_once_with(
        workspace.model_path,
        is_main_process=is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict.return_value,
    )
    if is_main_process:
        tokenizer.save_pretrained.assert_called_once_with(workspace.model_path)
        metadata = json.loads((workspace.model_path / "tune_config.json").read_text())
        assert metadata["seed"] == 710019
        assert metadata["best_hyperparameters"] == best_hparams
        assert metadata["config"]["head"]["head_type"] == "mlp"
    else:
        tokenizer.save_pretrained.assert_not_called()
        assert not (workspace.model_path / "tune_config.json").exists()
    accelerator.wait_for_everyone.assert_called_once()


class TestVarianceSeeds:
    """variance_seeds is a diagnostic: it must not change the returned model,
    only log validation-metric mean/std across extra training seeds."""

    def test_torch_probe_variance_diagnostic_restores_canonical_weights(
        self, monkeypatch, tmp_path, model_family, caplog
    ):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(
            task_type="sequence_classification",
            model_path=tmp_path / "model",
            num_labels=2,
        )
        step = fitting_state(
            TuneConfig(
                head=HeadConfig(head_type="mlp"),
                max_epochs=1,
                early_stopping_patience=None,
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )
        trained = fit_inputs(
            step.config,
            workspace,
            artifacts,
            seed=42,
            mode="finetune" if step.train_trunk else "tune_probe",
            publish=False,
            state=step,
        )
        canonical_state = {
            name: tensor.clone()
            for name, tensor in trained.predictor.model.state_dict().items()
        }
        accelerator = Accelerator()
        with caplog.at_level("INFO"):
            _report_torch_seed_variance(
                step.config,
                step.train_trunk,
                step.result,
                workspace,
                trained.predictor,
                trained.training,
                accelerator,
                {"learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 1},
            )
        assert "Seed variance (2 extra seeds" in caplog.text
        for name, tensor in trained.predictor.model.state_dict().items():
            assert torch.equal(tensor, canonical_state[name])

    def test_classical_head_logs_variance(
        self, monkeypatch, tmp_path, model_family, caplog
    ):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(
            task_type="sequence_classification",
            model_path=tmp_path / "model",
            num_labels=2,
        )
        step = fitting_state(
            TuneConfig(
                head=HeadConfig(head_type="sklearn_linear"),
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )
        with caplog.at_level("INFO"):
            fit_inputs(
                step.config,
                workspace,
                artifacts,
                seed=42,
                mode="finetune" if step.train_trunk else "tune_probe",
                publish=False,
                state=step,
            )
        assert "Seed variance (2 extra seeds" in caplog.text


class TestBohbAcrossTaskTypesAndSettings:
    """Exercise one representative gradient-path BOHB run end to end."""

    def test_bohb_trains_successfully(self, monkeypatch, tmp_path, model_family):
        pytest.importorskip("optuna")
        task_type = "sequence_classification"
        embedding_cache_enabled = True
        head_type = "mlp"
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            task_type,
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=embedding_cache_enabled),
        )
        workspace = MagicMock(
            task_type=task_type, model_path=tmp_path / "model", num_labels=2
        )
        step = fitting_state(
            TuneConfig(
                head=HeadConfig(head_type=head_type),
                max_epochs=1,
                accumulation_steps=2,
                save_best_model=False,
                hyperparameter_search=HyperparameterSearchConfig(
                    enabled=True,
                    n_trials=2,
                    search_space=HyperparameterSearchSpaceConfig(
                        learning_rate=[0.001, 0.01], weight_decay=[0.0], max_epochs=[1]
                    ),
                ),
            )
        )
        result = fit_inputs(
            step.config,
            workspace,
            artifacts,
            seed=42,
            mode="finetune" if step.train_trunk else "tune_probe",
            publish=False,
            state=step,
        )
        assert result.predictor.model is not None


_CLASSICAL_HEAD_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "knn": {"n_neighbors": [1, 3]}
}


class TestBohbAcrossClassicalHeadTypes:
    """Exercise one representative classical-head BOHB run."""

    def test_bohb_trains_successfully(self, monkeypatch, tmp_path, model_family):
        pytest.importorskip("optuna")
        task_type = "sequence_classification"
        head_type = "knn"
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            task_type,
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )
        workspace = MagicMock(
            task_type=task_type, model_path=tmp_path / "model", num_labels=2
        )
        step = fitting_state(
            TuneConfig(
                head=HeadConfig(
                    head_type=head_type,
                    search_space=_CLASSICAL_HEAD_SEARCH_SPACES[head_type],
                ),
                save_best_model=False,
                hyperparameter_search=HyperparameterSearchConfig(
                    enabled=True, n_trials=2, search_pattern="bohb"
                ),
            )
        )
        result = fit_inputs(
            step.config,
            workspace,
            artifacts,
            seed=42,
            mode="finetune" if step.train_trunk else "tune_probe",
            publish=False,
            state=step,
        )
        assert hasattr(result.predictor.model, "predict")


@pytest.mark.parametrize("monitor", [False, True])
def test_cached_validation_features_are_prepared_once_per_training_run(
    monkeypatch, tmp_path, model_family, monitor
):
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        embedding_cache=EmbeddingCacheConfig(enabled=True),
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    accelerator = Accelerator(cpu=True)
    step = fitting_state(
        TuneConfig(
            max_epochs=3,
            early_stopping_patience=5 if monitor else None,
            save_best_model=False,
        )
    )
    prepare_split = MagicMock(wraps=_cached_embedding_split)
    evaluate_validation = evaluate_torch_validation
    validation_calls = MagicMock(wraps=evaluate_validation)
    patch_function(monkeypatch, _cached_embedding_split, prepare_split)
    patch_function(monkeypatch, evaluate_torch_validation, validation_calls)
    hparams = {"learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 3}
    previous_cached_val = None
    for _ in range(2):
        prepare_split.reset_mock()
        validation_calls.reset_mock()
        score = train_torch(
            step.config,
            step.train_trunk,
            workspace,
            prepared.predictor,
            prepared.training,
            accelerator,
            hparams,
        )
        validation_preparations = [
            call
            for call in prepare_split.call_args_list
            if call.args[0] is prepared.training.val_dataloader
        ]
        assert len(validation_preparations) == (1 if monitor else 0)
        assert validation_calls.call_count == (3 if monitor else 0)
        if monitor:
            cached_val = validation_calls.call_args.kwargs["cached_val"]
            assert cached_val is not None
            assert cached_val is not previous_cached_val
            assert all(
                (
                    call.kwargs["cached_val"] is cached_val
                    for call in validation_calls.call_args_list
                )
            )
            previous_cached_val = cached_val
            validation_forward = MagicMock(wraps=prepared.predictor.model.forward)
            with monkeypatch.context() as validation:
                validation.setattr(
                    prepared.predictor.model, "forward", validation_forward
                )
                assert score == evaluate_torch_validation(
                    step.config,
                    workspace,
                    prepared.predictor,
                    prepared.training,
                    accelerator,
                    resolve_selection_metric(step.config, workspace),
                )
            assert validation_forward.called
            assert all(
                (
                    set(call.kwargs) == {"inputs_embeds"}
                    for call in validation_forward.call_args_list
                )
            )
        else:
            assert score is None


def test_cached_training_features_are_transferred_only_when_training(
    monkeypatch, tmp_path, model_family
):
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        embedding_cache=EmbeddingCacheConfig(enabled=True),
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    accelerator = Accelerator(cpu=True)
    step = fitting_state(
        TuneConfig(max_epochs=1, early_stopping_patience=None, save_best_model=False)
    )
    prepare_split = MagicMock(wraps=_cached_embedding_split)
    patch_function(monkeypatch, _cached_embedding_split, prepare_split)
    result = fit_inputs(
        step.config,
        workspace,
        prepared,
        accelerator=accelerator,
        mode="finetune" if step.train_trunk else "tune_probe",
        publish=False,
        state=step,
    )
    assert result is prepared
    prepare_split.assert_called_once_with(prepared.training.train_dataloader)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "batch_size, train_batch_sizes, val_batch_sizes",
    [(None, [4, 4, 3], [5, 5, 1]), (3, [3, 3, 3, 2], [3, 3, 3, 2])],
)
def test_cached_probe_batch_size_honors_loader_and_override(
    monkeypatch, wrapped, batch_size, train_batch_sizes, val_batch_sizes
):
    accelerator = Accelerator(cpu=True)
    dataset = EmbeddingDataset(
        torch.randn(11, 4), list(range(11)), torch.arange(11) % 2
    )
    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=5)
    if wrapped:
        train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
        assert train_loader.batch_size is None
        assert train_loader.batch_sampler.batch_size == 4
    model = EmbeddingProbeHead(4, 2, "sequence_classification")
    prepared = make_inputs(
        model=model,
        tokenizer=MagicMock(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=None,
    )
    step = fitting_state(
        TuneConfig(
            batch_size=batch_size,
            max_epochs=2,
            early_stopping_patience=None,
            save_best_model=False,
        )
    )
    optimizer_steps = []

    def track_optimizer(*args, **kwargs):
        optimizer = build_optimizer(*args, **kwargs)
        optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(None))
        return optimizer

    scheduler = MagicMock(wraps=get_scheduler)
    forward = MagicMock(wraps=model.forward)
    monkeypatch.setattr(
        "modules.evaluate.src.trainers.torch.build_optimizer", track_optimizer
    )
    monkeypatch.setattr("modules.evaluate.src.trainers.torch.get_scheduler", scheduler)
    monkeypatch.setattr(model, "forward", forward)
    workspace = MagicMock(task_type="sequence_classification", num_labels=2)
    hparams = {"learning_rate": 0.001, "weight_decay": 0.0, "max_epochs": 2}
    train_torch(
        step.config,
        step.train_trunk,
        workspace,
        prepared.predictor,
        prepared.training,
        accelerator,
        hparams,
    )
    assert [
        call.kwargs["inputs_embeds"].shape[0] for call in forward.call_args_list
    ] == train_batch_sizes * 2
    expected_steps = len(train_batch_sizes) * 2
    assert len(optimizer_steps) == expected_steps
    assert scheduler.call_args.kwargs["num_training_steps"] == expected_steps
    forward.reset_mock()
    evaluate_torch_validation(
        step.config, workspace, prepared.predictor, prepared.training, accelerator, "f1"
    )
    assert [
        call.kwargs["inputs_embeds"].shape[0] for call in forward.call_args_list
    ] == val_batch_sizes


def test_cached_mlp_checkpoint_reloads_trained_head(
    monkeypatch, tmp_path, model_family
):
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        "sequence_classification",
        model_family,
        embedding_cache=EmbeddingCacheConfig(enabled=True),
    )
    workspace = _make_workspace(tmp_path, "sequence_classification")
    step = fitting_state(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=7),
            learning_rate=0.01,
            max_epochs=1,
            warmup_ratio=0,
            early_stopping_patience=None,
        )
    )
    trained = fit_inputs(
        step.config,
        workspace,
        prepared,
        accelerator=Accelerator(cpu=True),
        mode="finetune" if step.train_trunk else "tune_probe",
    )
    metadata = json.loads((workspace.model_path / "probe_head_config.json").read_text())
    assert metadata["probe_hidden_size"] == 7
    assert metadata["standardize_features"] is True
    reloaded = EmbeddingProbeHead(
        metadata["embedding_size"],
        metadata["num_labels"],
        metadata["task_type"],
        probe_hidden_size=metadata["probe_hidden_size"],
        standardize_features=metadata["standardize_features"],
    )
    state = torch.load(workspace.model_path / "probe_head.pt", weights_only=True)
    reloaded.load_state_dict(state, strict=True)
    for name, tensor in trained.predictor.model.state_dict().items():
        torch.testing.assert_close(state[name], tensor)
    features = prepared.training.val_dataloader.dataset.embeddings
    with torch.inference_mode():
        torch.testing.assert_close(
            reloaded(inputs_embeds=features).logits,
            trained.predictor.model(inputs_embeds=features).logits,
        )
