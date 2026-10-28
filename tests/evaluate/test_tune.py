import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.config import LayerwiseConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.steps.tune import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
    TuneStep,
    _improves,
)
from modules.evaluate.src.artifacts.feature_cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
)
from modules.evaluate.src.model.heads import EmbeddingProbeHead, HeadConfig
from modules.evaluate.src.utils.hpo import (
    build_optimizer,
    build_scheduler,
    checkpoint_state,
    resolve_search_direction,
    sample_candidates,
    split_decay_parameters,
)
from modules.evaluate.src.predictors.embeddings import EmbeddingDataset

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
    prepared = PreparedArtifacts(
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
        "modules.evaluate.src.steps.tune.run_inference",
        lambda *args, **kwargs: ([], live_labels, live_labels, None, None),
    )
    step = TuneStep(TuneConfig(metric_aggregation="macro"))
    score = step._evaluate_validation(
        _make_workspace(tmp_path, task_type, num_labels=num_labels),
        prepared,
        Accelerator(cpu=True),
        "f1",
    )
    assert score == pytest.approx(1 / num_labels)


def test_classical_selection_uses_fixed_class_macro_f1(monkeypatch):
    from modules.evaluate.src.model.heads import sweep_classical_head

    candidates = {
        "observed_winner": np.zeros(8, dtype=int),
        "fixed_winner": np.array([0, 2, 2, 2, 1, 2, 2, 2]),
    }

    def build(head_type, task_type, hparams, **kwargs):
        return SimpleNamespace(
            fit=lambda *args: None,
            predict=lambda *args: candidates[hparams["candidate"]],
        )

    monkeypatch.setattr("modules.evaluate.src.model.heads.build_estimator", build)
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
            enabled=True,
            layers=[0, -1],
            autocast_dtype="float32",
        ),
    )
    assert isinstance(prepared.model, LayerwiseProbeModel)
    workspace_config = _make_workspace(tmp_path, "sequence_classification").config
    workspace = get_evaluation_workspace(workspace_config, variant="layerwise-test")
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type="torch_linear"),
            max_epochs=1,
            early_stopping_patience=None,
        )
    )
    trained = step.run(workspace, prepared, accelerator=Accelerator(cpu=True))
    assert isinstance(trained.model, LayerwiseProbeModel)
    assert all(parameter.grad is None for parameter in trained.model.trunk.parameters())
    for layer in (0, 2):
        layer_workspace = get_evaluation_workspace(
            workspace_config, variant=f"layerwise-test-layer_{layer}"
        )
        checkpoint = torch.load(
            layer_workspace.model_path / "probe_head.pt", weights_only=True
        )
        assert checkpoint["layer"] == layer
        assert checkpoint["layerwise_config"]["layers"] == [0, -1]
        assert checkpoint["selection"]["epoch"] == 1
        assert checkpoint["feature_std"].shape == (trained.model.embedding_size,)


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
    assert _improves(score, best, direction) is expected


def test_checkpoint_state_is_independent_cpu_copy() -> None:
    model = nn.Linear(2, 1)

    checkpoint = checkpoint_state(model)

    assert all(tensor.device.type == "cpu" for tensor in checkpoint.values())
    with torch.no_grad():
        model.weight.add_(1)
    assert not torch.equal(checkpoint["weight"], model.weight.cpu())


@pytest.mark.parametrize(
    "norm_type", ["RMSNorm", "CenteredRMSNorm", "ZeroCenteredRMSNorm", "LayerNorm"]
)
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
        learning_rate=[1e-4, 1e-3],
        weight_decay=[0.0],
        max_epochs=[1, 2],
    )
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(search_space=search_space)
        )
    )

    candidates = step._generate_candidates("random", n_trials=10, seed=42)

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
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(
                enabled=True,
                n_trials=4,
                search_space=HyperparameterSearchSpaceConfig(
                    learning_rate=[1e-4],
                    weight_decay=[1e-6, 1e-1],
                    weight_decay_sampling="log",
                    max_epochs=[1],
                ),
            ),
        )
    )
    samples = []

    def train(workspace, artifacts, accelerator, hparams, **kwargs):
        distribution = kwargs["trial"].distributions["weight_decay"]
        assert distribution.log
        assert distribution.low == 1e-6
        assert distribution.high == 1e-1
        samples.append(hparams["weight_decay"])
        return hparams["weight_decay"]

    monkeypatch.setattr(step, "_train", train)
    for _ in range(2):
        prepared = PreparedArtifacts(
            model=EmbeddingProbeHead(3, 2, "sequence_classification"),
            tokenizer=MagicMock(),
            train_dataloader=loader,
            val_dataloader=loader,
            test_dataloader=None,
        )
        best = step._search_hyperparameters(workspace, prepared, accelerator, seed=42)
        assert best["weight_decay"] == max(samples[-4:])

    assert samples[:4] == samples[4:]
    assert all(1e-6 < value < 1e-1 for value in samples)
    assert len(set(samples[:4])) > 1


def test_categorical_weight_decay_preserves_zero_and_grid_candidates():
    trial = MagicMock()
    trial.suggest_categorical.return_value = 0.0
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(
                search_space=HyperparameterSearchSpaceConfig(
                    learning_rate=[1e-4],
                    weight_decay=[0.0, 1e-3],
                    max_epochs=[1],
                )
            )
        )
    )
    assert step._suggest_weight_decay(trial) == 0.0
    trial.suggest_categorical.assert_called_once_with("weight_decay", [0.0, 1e-3])
    trial.suggest_float.assert_not_called()
    for pattern in ("random", "grid"):
        candidates = step._generate_candidates(pattern, n_trials=10, seed=42)
        assert {candidate["weight_decay"] for candidate in candidates} == {0.0, 1e-3}

    space = step.config.hyperparameter_search.search_space
    space.weight_decay = [1e-6, 1e-1]
    space.weight_decay_sampling = "log"
    for pattern in ("random", "grid"):
        candidates = step._generate_candidates(pattern, n_trials=10, seed=42)
        assert {candidate["weight_decay"] for candidate in candidates} == {1e-6, 1e-1}


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

    actual_embeddings, actual_labels = TuneStep._gather_embeddings(
        wrapped_loader, accelerator
    )

    np.testing.assert_array_equal(actual_embeddings, embeddings.numpy())
    np.testing.assert_array_equal(actual_labels, labels.numpy())
    accelerator.gather_for_metrics.assert_not_called()


def test_resolve_direction_flips_to_minimize_for_lower_is_better_metric() -> None:
    # "mse" (the sequence_regression default metric) is lower-is-better, so
    # an unset direction must resolve to minimize.
    step = TuneStep(TuneConfig())

    assert step._resolve_direction("mse") == "minimize"
    assert step._resolve_direction("f1") == "maximize"


def test_resolve_direction_respects_explicit_direction() -> None:
    # An explicitly-set direction (even one that seems mismatched) must be
    # honored rather than overridden.
    step = TuneStep(
        TuneConfig(
            hyperparameter_search=HyperparameterSearchConfig(direction="maximize")
        )
    )

    assert step._resolve_direction("mse") == "maximize"


def test_resolve_search_direction_uses_metric_default_unless_explicit() -> None:
    assert resolve_search_direction("mse", None) == "minimize"
    assert resolve_search_direction("mse", "maximize") == "maximize"


def test_save_model_persists_model_and_tokenizer_on_main_process(tmp_path) -> None:
    step = TuneStep(TuneConfig())
    workspace = MagicMock(model_path=tmp_path / "model")

    model = MagicMock()
    tokenizer = MagicMock()
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    )

    accelerator = MagicMock(is_main_process=True)
    accelerator.unwrap_model.return_value = model

    best_hparams = {"learning_rate": 1e-4, "weight_decay": 1e-4, "max_epochs": 10}
    step._save_model(workspace, prepared, model, accelerator, best_hparams, seed=710019)

    assert workspace.model_path.is_dir()
    accelerator.unwrap_model.assert_called_once_with(model)
    model.save_pretrained.assert_called_once_with(
        workspace.model_path,
        is_main_process=True,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict.return_value,
    )
    tokenizer.save_pretrained.assert_called_once_with(workspace.model_path)
    accelerator.wait_for_everyone.assert_called_once()

    tune_config_path = workspace.model_path / "tune_config.json"
    assert tune_config_path.exists()
    on_disk = json.loads(tune_config_path.read_text())
    assert on_disk["seed"] == 710019
    assert on_disk["best_hyperparameters"] == best_hparams
    assert on_disk["config"]["head"]["head_type"] == "mlp"


def test_save_model_skips_tokenizer_save_on_non_main_process(tmp_path) -> None:
    step = TuneStep(TuneConfig())
    workspace = MagicMock(model_path=tmp_path / "model")

    model = MagicMock()
    tokenizer = MagicMock()
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    )

    accelerator = MagicMock(is_main_process=False)
    accelerator.unwrap_model.return_value = model

    best_hparams = {"learning_rate": 1e-4, "weight_decay": 1e-4, "max_epochs": 10}
    step._save_model(workspace, prepared, model, accelerator, best_hparams, seed=710019)

    model.save_pretrained.assert_called_once_with(
        workspace.model_path,
        is_main_process=False,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict.return_value,
    )
    tokenizer.save_pretrained.assert_not_called()
    accelerator.wait_for_everyone.assert_called_once()
    assert not (workspace.model_path / "tune_config.json").exists()


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
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="mlp"),
                max_epochs=1,
                early_stopping_patience=None,
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )
        trained = step.run(workspace, artifacts, seed=42)
        canonical_state = {
            name: tensor.clone() for name, tensor in trained.model.state_dict().items()
        }

        accelerator = Accelerator()
        with caplog.at_level("INFO"):
            step._report_torch_seed_variance(
                workspace,
                trained,
                accelerator,
                {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 1},
            )

        assert "Seed variance (2 extra seeds" in caplog.text
        for name, tensor in trained.model.state_dict().items():
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
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type="sklearn_linear"),
                save_best_model=False,
                variance_seeds=[1, 2],
            )
        )

        with caplog.at_level("INFO"):
            step.run(workspace, artifacts, seed=42)

        assert "Seed variance (2 extra seeds" in caplog.text


class TestBohbAcrossTaskTypesAndSettings:
    """Exercise one representative gradient-path BOHB run end to end."""

    def test_bohb_trains_successfully(
        self,
        monkeypatch,
        tmp_path,
        model_family,
    ):
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
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(head_type=head_type),
                max_epochs=1,
                accumulation_steps=2,
                save_best_model=False,
                hyperparameter_search=HyperparameterSearchConfig(
                    enabled=True,
                    n_trials=2,
                    search_space=HyperparameterSearchSpaceConfig(
                        learning_rate=[1e-3, 1e-2],
                        weight_decay=[0.0],
                        max_epochs=[1],
                    ),
                ),
            )
        )

        result = step.run(workspace, artifacts, seed=42)

        assert result.model is not None


# Search spaces small enough to run quickly under bohb's real Optuna study,
# but with >1 candidate value so bohb actually has something to search over
# (sweep_classical_head's bohb branch is a no-op without one -- see heads.py).
_CLASSICAL_HEAD_SEARCH_SPACES: dict[str, dict[str, list]] = {
    "knn": {"n_neighbors": [1, 3]},
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
        step = TuneStep(
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

        result = step.run(workspace, artifacts, seed=42)

        assert hasattr(result.model, "predict")


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
    step = TuneStep(
        TuneConfig(
            max_epochs=3,
            early_stopping_patience=5 if monitor else None,
            save_best_model=False,
        )
    )
    prepare_split = MagicMock(wraps=step._cached_embedding_split)
    evaluate_validation = step._evaluate_validation
    validation_calls = MagicMock(wraps=evaluate_validation)
    monkeypatch.setattr(step, "_cached_embedding_split", prepare_split)
    monkeypatch.setattr(step, "_evaluate_validation", validation_calls)
    hparams = {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 3}
    previous_cached_val = None

    for _ in range(2):
        prepare_split.reset_mock()
        validation_calls.reset_mock()
        score = step._train(workspace, prepared, accelerator, hparams)

        validation_preparations = [
            call
            for call in prepare_split.call_args_list
            if call.args[0] is prepared.val_dataloader
        ]
        assert len(validation_preparations) == (1 if monitor else 0)
        assert validation_calls.call_count == (3 if monitor else 0)
        if monitor:
            cached_val = validation_calls.call_args.kwargs["cached_val"]
            assert cached_val is not None
            assert cached_val is not previous_cached_val
            assert all(
                call.kwargs["cached_val"] is cached_val
                for call in validation_calls.call_args_list
            )
            previous_cached_val = cached_val
            # Standalone validation still prepares its own features and
            # agrees with the best score restored by the training run.
            validation_forward = MagicMock(wraps=prepared.model.forward)
            with monkeypatch.context() as validation:
                validation.setattr(prepared.model, "forward", validation_forward)
                assert score == evaluate_validation(
                    workspace,
                    prepared,
                    accelerator,
                    step._resolve_metric_name(workspace),
                )
            assert validation_forward.called
            assert all(
                set(call.kwargs) == {"inputs_embeds"}
                for call in validation_forward.call_args_list
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
    step = TuneStep(
        TuneConfig(
            max_epochs=1,
            early_stopping_patience=None,
            save_best_model=False,
        )
    )
    prepare_split = MagicMock(wraps=step._cached_embedding_split)
    monkeypatch.setattr(step, "_cached_embedding_split", prepare_split)

    result = step.run(workspace, prepared, accelerator=accelerator)

    assert result is prepared
    prepare_split.assert_called_once_with(prepared.train_dataloader)


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
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=MagicMock(),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=None,
    )
    step = TuneStep(
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

    scheduler = MagicMock(wraps=build_scheduler)
    forward = MagicMock(wraps=model.forward)
    monkeypatch.setattr(
        "modules.evaluate.src.steps.tune.build_optimizer", track_optimizer
    )
    monkeypatch.setattr("modules.evaluate.src.steps.tune.build_scheduler", scheduler)
    monkeypatch.setattr(model, "forward", forward)
    workspace = MagicMock(task_type="sequence_classification", num_labels=2)
    hparams = {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 2}

    step._train(workspace, prepared, accelerator, hparams)

    assert [
        call.kwargs["inputs_embeds"].shape[0] for call in forward.call_args_list
    ] == train_batch_sizes * 2
    expected_steps = len(train_batch_sizes) * 2
    assert len(optimizer_steps) == expected_steps
    assert scheduler.call_args.kwargs["num_training_steps"] == expected_steps

    forward.reset_mock()
    step._evaluate_validation(workspace, prepared, accelerator, "f1")
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
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=7),
            learning_rate=1e-2,
            max_epochs=1,
            warmup_ratio=0,
            early_stopping_patience=None,
        )
    )

    trained = step.run(workspace, prepared, accelerator=Accelerator(cpu=True))

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
    for name, tensor in trained.model.state_dict().items():
        torch.testing.assert_close(state[name], tensor)
    features, _ = step._cached_embedding_split(prepared.val_dataloader)
    with torch.inference_mode():
        torch.testing.assert_close(
            reloaded(inputs_embeds=features).logits,
            trained.model(inputs_embeds=features).logits,
        )
