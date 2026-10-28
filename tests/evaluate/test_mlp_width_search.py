"""Cached MLP width searches preserve trial initialization and the winning head."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from pydantic import ValidationError
from torch.utils.data import DataLoader

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.model.heads import EmbeddingProbeHead, HeadConfig
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.tune import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
    TuneStep,
)
from modules.evaluate.src.predictors.embeddings import EmbeddingDataset

from .test_prepare import _make_workspace

_FEATURES = torch.tensor(
    [[2.0, 10.0, 7.0], [4.0, 14.0, 7.0], [6.0, 18.0, 7.0], [8.0, 22.0, 7.0]]
)
_WIDTHS = [128, 256, 512, 1024]


def _prepared():
    def loader(features):
        return DataLoader(
            EmbeddingDataset(features, list(range(4)), torch.tensor([0, 1, 0, 1])),
            batch_size=2,
            shuffle=True,
            generator=torch.Generator().manual_seed(42),
        )

    return PreparedArtifacts(
        model=EmbeddingProbeHead(3, 2, "sequence_classification"),
        tokenizer=MagicMock(),
        train_dataloader=loader(_FEATURES.clone()),
        val_dataloader=loader(_FEATURES + 1000),
        test_dataloader=None,
    )


def _search(pattern, *, widths=None, depths=None, n_trials=24):
    return HyperparameterSearchConfig(
        enabled=True,
        search_pattern=pattern,
        n_trials=n_trials,
        search_space=HyperparameterSearchSpaceConfig(
            hidden_size=widths,
            hidden_depth=depths,
            learning_rate=[1e-3, 1e-2],
            weight_decay=[0.0],
            max_epochs=[1],
        ),
    )


@pytest.mark.parametrize("widths", [[], [0], [-1], [128, 0]])
def test_width_search_rejects_empty_or_nonpositive_widths(widths):
    with pytest.raises(ValidationError, match="hidden_size"):
        HyperparameterSearchSpaceConfig(hidden_size=widths)


@pytest.mark.parametrize("depths", [[], [0], [-1], [1, 0]])
def test_depth_search_rejects_empty_or_nonpositive_depths(depths):
    with pytest.raises(ValidationError, match="hidden_depth"):
        HyperparameterSearchSpaceConfig(hidden_depth=depths)


def test_grid_width_candidates_cover_the_full_product():
    step = TuneStep(TuneConfig(hyperparameter_search=_search("grid", widths=_WIDTHS)))

    candidates = step._generate_candidates("grid", n_trials=1, seed=42)

    assert len(candidates) == 8
    assert {(item["hidden_size"], item["learning_rate"]) for item in candidates} == {
        (width, rate) for width in _WIDTHS for rate in [1e-3, 1e-2]
    }
    assert all(item["seed"] == 42 for item in candidates)


def test_random_width_candidates_cover_each_width_with_a_limited_budget():
    step = TuneStep(TuneConfig(hyperparameter_search=_search("random", widths=_WIDTHS)))

    candidates = step._generate_candidates("random", n_trials=4, seed=42)

    assert len(candidates) == 4
    assert {item["hidden_size"] for item in candidates} == set(_WIDTHS)
    assert len({tuple(sorted(item.items())) for item in candidates}) == 4
    assert candidates == step._generate_candidates("random", n_trials=4, seed=42)


def test_grid_combines_width_depth_and_batch_dimensions():
    search = _search("grid", widths=[3, 5], depths=[1, 2])
    search.search_space.learning_rate = [1e-3]
    search.search_space.batch_size = [2, 4]
    step = TuneStep(TuneConfig(hyperparameter_search=search))
    candidates = step._generate_candidates("grid", n_trials=1, seed=42)
    assert {
        (item["hidden_size"], item["hidden_depth"], item["batch_size"])
        for item in candidates
    } == {
        (width, depth, batch)
        for width in [3, 5]
        for depth in [1, 2]
        for batch in [2, 4]
    }
    assert len(candidates) == 8


@pytest.mark.parametrize("pattern", ["grid", "random"])
def test_fixed_width_candidates_keep_existing_hyperparameters(pattern):
    step = TuneStep(TuneConfig(hyperparameter_search=_search(pattern)))

    candidates = step._generate_candidates(pattern, n_trials=24, seed=42)

    assert len(candidates) == 2
    assert all(
        set(item) == {"learning_rate", "weight_decay", "max_epochs", "seed"}
        for item in candidates
    )


@pytest.mark.parametrize("pattern", ["grid", "random", "bohb"])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("depths", [None, [1, 2]])
def test_width_search_saves_winning_architecture_weights_and_training_statistics(
    monkeypatch, tmp_path, pattern, standardize, depths
):
    if pattern == "bohb":
        pytest.importorskip("optuna")
    prepared = _prepared()
    workspace = _make_workspace(tmp_path, "sequence_classification")
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=7),
            standardize_features=standardize,
            hyperparameter_search=_search(pattern, widths=_WIDTHS, depths=depths),
        )
    )
    trials = []

    def train(workspace, artifacts, accelerator, hparams, **kwargs):
        model = artifacts.model
        width = hparams["hidden_size"]
        depth = hparams.get("hidden_depth", 1)
        assert model.probe_hidden_size == width
        assert model.probe_hidden_depth == depth
        assert (
            sum(isinstance(layer, torch.nn.Linear) for layer in model.projection)
            == depth
        )
        initial = {name: value.clone() for name, value in model.state_dict().items()}
        if standardize:
            torch.testing.assert_close(model.feature_mean, _FEATURES.mean(dim=0))
            torch.testing.assert_close(
                model.feature_std, _FEATURES.std(dim=0, correction=0).clamp_min(1e-6)
            )
        else:
            assert model.feature_mean is None
            assert model.feature_std is None
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_((len(trials) + 1) / 100)
        score = -float(width) + len(trials) * 1e-5
        trials.append(
            {
                "width": width,
                "depth": depth,
                "initial": initial,
                "global_order": torch.randperm(6),
                "loader_order": torch.randperm(
                    6, generator=artifacts.train_dataloader.generator
                ),
                "trained": {
                    name: value.clone() for name, value in model.state_dict().items()
                },
                "score": score,
            }
        )
        return score

    monkeypatch.setattr(step, "_train", train)

    result = step.run(workspace, prepared, seed=42, accelerator=Accelerator(cpu=True))

    assert {trial["width"] for trial in trials} == set(_WIDTHS)
    assert len(trials) == (24 if pattern == "bohb" else 8 * len(depths or [1]))
    assert {trial["depth"] for trial in trials} == set(depths or [1])
    if pattern in {"random", "bohb"}:
        assert {trial["width"] for trial in trials[:4]} == set(_WIDTHS)
    for trial in trials[1:]:
        torch.testing.assert_close(trial["global_order"], trials[0]["global_order"])
        torch.testing.assert_close(trial["loader_order"], trials[0]["loader_order"])
    for width in _WIDTHS:
        for depth in depths or [1]:
            starts = [
                trial["initial"]
                for trial in trials
                if (trial["width"], trial["depth"]) == (width, depth)
            ]
            for state in starts[1:]:
                for name, value in starts[0].items():
                    torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    best = max(trials, key=lambda trial: trial["score"])
    assert result.model.probe_hidden_size == best["width"]
    assert result.model.probe_hidden_depth == best["depth"]
    for name, value in best["trained"].items():
        torch.testing.assert_close(result.model.state_dict()[name], value)

    metadata = json.loads((workspace.model_path / "probe_head_config.json").read_text())
    assert metadata["probe_hidden_size"] == best["width"]
    assert metadata["probe_hidden_depth"] == best["depth"]
    assert metadata["best_hyperparameters"]["hidden_size"] == best["width"]
    restored = EmbeddingProbeHead(
        metadata["embedding_size"],
        metadata["num_labels"],
        metadata["task_type"],
        probe_hidden_size=metadata["probe_hidden_size"],
        probe_hidden_depth=metadata["probe_hidden_depth"],
        standardize_features=metadata["standardize_features"],
    )
    restored.load_state_dict(
        torch.load(workspace.model_path / "probe_head.pt", weights_only=True),
        strict=True,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(inputs_embeds=_FEATURES + 1000).logits,
            result.model(inputs_embeds=_FEATURES + 1000).logits,
        )


@pytest.mark.parametrize(
    "unsupported", ["torch_linear", "uncached", "finetune", "layerwise"]
)
@pytest.mark.parametrize("dimension", ["hidden_size", "hidden_depth"])
def test_width_search_rejects_unsupported_probe_paths(tmp_path, unsupported, dimension):
    prepared = _prepared()
    if unsupported == "uncached":
        prepared.train_dataloader = DataLoader([{"labels": 0}], batch_size=1)
    elif unsupported == "layerwise":
        prepared.model = MagicMock(spec=LayerwiseProbeModel)

    with pytest.raises(
        ValueError,
        match=f"{dimension} search requires a cached MLP probe in tune_probe mode",
    ):
        step = TuneStep(
            TuneConfig(
                head=HeadConfig(
                    head_type="torch_linear" if unsupported == "torch_linear" else "mlp"
                ),
                hyperparameter_search=_search(
                    "grid",
                    widths=[3, 5] if dimension == "hidden_size" else None,
                    depths=[1, 2] if dimension == "hidden_depth" else None,
                ),
            ),
            mode="finetune" if unsupported == "finetune" else "tune_probe",
        )
        step.run(
            _make_workspace(tmp_path, "sequence_classification"),
            prepared,
            accelerator=Accelerator(cpu=True),
        )


def test_disabled_width_search_keeps_the_configured_fixed_width(monkeypatch, tmp_path):
    search = _search("grid", widths=[3, 5])
    search.enabled = False
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=7, hidden_depth=2),
            hyperparameter_search=search,
            save_best_model=False,
        )
    )

    def train(workspace, artifacts, accelerator, hparams):
        assert artifacts.model.probe_hidden_size == 7
        assert artifacts.model.probe_hidden_depth == 2
        assert "hidden_size" not in hparams

    monkeypatch.setattr(step, "_train", train)

    trained = step.run(
        _make_workspace(tmp_path, "sequence_classification"),
        _prepared(),
        accelerator=Accelerator(cpu=True),
    )

    assert trained.model.probe_hidden_size == 7
    assert trained.model.probe_hidden_depth == 2


def test_different_depths_do_not_share_control_cache_identity():
    shallow = EmbeddingProbeHead(
        3, 2, "sequence_classification", 7, probe_hidden_depth=1
    )
    deep = EmbeddingProbeHead(3, 2, "sequence_classification", 7, probe_hidden_depth=2)
    assert shallow.cache_identity() != deep.cache_identity()


@pytest.mark.parametrize("pattern", ["grid", "random", "bohb"])
def test_depth_only_search_trains_every_depth_and_keeps_fixed_width(
    monkeypatch, tmp_path, pattern
):
    search = _search(pattern, depths=[1, 2], n_trials=4)
    search.search_space.learning_rate = [1e-3]
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(hidden_size=7),
            hyperparameter_search=search,
            early_stopping_patience=None,
            save_best_model=False,
        )
    )
    trained_depths = []
    train = step._train

    def record_training(workspace, artifacts, accelerator, hparams, **kwargs):
        model = artifacts.model
        assert model.probe_hidden_size == 7
        before = [
            layer.weight.detach().clone()
            for layer in model.projection
            if isinstance(layer, torch.nn.Linear)
        ]
        score = train(workspace, artifacts, accelerator, hparams, **kwargs)
        after = [
            layer.weight
            for layer in model.projection
            if isinstance(layer, torch.nn.Linear)
        ]
        assert all(not torch.equal(old, new) for old, new in zip(before, after))
        trained_depths.append(model.probe_hidden_depth)
        return score

    monkeypatch.setattr(step, "_train", record_training)
    result = step.run(
        _make_workspace(tmp_path, "sequence_classification"),
        _prepared(),
        accelerator=Accelerator(cpu=True),
        seed=42,
    )
    assert set(trained_depths) == {1, 2}
    assert result.model.probe_hidden_depth in trained_depths
    assert result.model.probe_hidden_size == 7


def test_width_search_trains_cached_heads_end_to_end(monkeypatch, tmp_path):
    search = _search("grid", widths=[3, 5], n_trials=2)
    search.search_space.learning_rate = [1e-3]
    step = TuneStep(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=7),
            hyperparameter_search=search,
            early_stopping_patience=None,
            save_best_model=False,
        )
    )
    trained_widths = []
    train = step._train

    def record_training(workspace, artifacts, accelerator, hparams, **kwargs):
        model = artifacts.model
        before = model.classifier.weight.detach().clone()
        result = train(workspace, artifacts, accelerator, hparams, **kwargs)
        assert not torch.equal(model.classifier.weight, before)
        trained_widths.append(model.probe_hidden_size)
        return result

    monkeypatch.setattr(step, "_train", record_training)

    result = step.run(
        _make_workspace(tmp_path, "sequence_classification"),
        _prepared(),
        accelerator=Accelerator(cpu=True),
        seed=42,
    )

    assert set(trained_widths) == {3, 5}
    assert result.model.probe_hidden_size in trained_widths
    with torch.inference_mode():
        assert torch.isfinite(result.model(inputs_embeds=_FEATURES + 1000).logits).all()


@pytest.mark.parametrize(
    "override",
    [
        None,
        "steps.tune.head.head_type=torch_linear",
        "steps.prepare.embedding_cache.enabled=false",
        "steps.mode=finetune",
    ],
)
def test_default_mlp_width_search_validates_supported_configuration(override):
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    paths = [
        root / "config.yaml",
        root / "tasks/sequence_classification.yaml",
        root / "datasets/localization_prediction.yaml",
    ]
    if override is not None:
        overrides = [override]
        if override == "steps.mode=finetune":
            overrides.append("steps.prepare.embedding_cache.enabled=false")
        with pytest.raises(
            ValueError, match="hidden_size search requires a cached MLP probe"
        ):
            load_and_parse(path=paths, model_cls=EvaluationConfig, overrides=overrides)
        return

    config = load_and_parse(path=paths, model_cls=EvaluationConfig)

    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.n_trials == 100
    assert search.search_space.hidden_size == [128, 256, 512]
    assert search.search_space.hidden_depth == [1, 2]


@pytest.mark.parametrize(
    "overrides",
    [
        ["steps.tune.head.head_type=torch_linear"],
        ["steps.prepare.embedding_cache.enabled=false"],
        ["steps.mode=finetune", "steps.prepare.embedding_cache.enabled=false"],
    ],
)
def test_depth_only_search_config_rejects_unsupported_paths(overrides):
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    with pytest.raises(
        ValueError, match="hidden_depth search requires a cached MLP probe"
    ):
        load_and_parse(
            path=[
                root / "config.yaml",
                root / "tasks/sequence_classification.yaml",
                root / "datasets/localization_prediction.yaml",
            ],
            model_cls=EvaluationConfig,
            overrides=[
                "steps.tune.hyperparameter_search.search_space.hidden_size=null",
                "steps.tune.hyperparameter_search.search_space.batch_size=null",
                *overrides,
            ],
        )
