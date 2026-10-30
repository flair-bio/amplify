"""Cached MLP searches apply each trial's batch size throughout training."""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from accelerate import Accelerator
from pydantic import ValidationError
from torch.utils.data import DataLoader

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.model.heads import EmbeddingProbeHead, HeadConfig
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.predictors.embeddings import EmbeddingDataset
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.training.config import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
)
from modules.evaluate.src.training.search import generate_candidates
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.training.tuning import fit_probe_protocol
from modules.evaluate.src.utils.config_loader import (
    load_and_parse_evaluation as load_and_parse,
)
from modules.evaluate.src.utils.hpo import build_optimizer, build_scheduler

from .test_prepare import _make_workspace

_BATCH_SIZES = [8, 16, 32, 64, 128]


def _prepared(train_rows=131, val_rows=137):
    def loader(rows):
        features = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        return DataLoader(
            EmbeddingDataset(
                features / 1000, list(range(rows)), torch.arange(rows) % 2
            ),
            batch_size=6,
        )

    return PreparedArtifacts(
        model=EmbeddingProbeHead(3, 2, "sequence_classification", probe_hidden_size=4),
        tokenizer=MagicMock(),
        train_dataloader=loader(train_rows),
        val_dataloader=loader(val_rows),
        test_dataloader=None,
    )


def _search(pattern, *, batch_sizes=None, widths=None, n_trials=10):
    return HyperparameterSearchConfig(
        enabled=True,
        search_pattern=pattern,
        n_trials=n_trials,
        search_space=HyperparameterSearchSpaceConfig(
            batch_size=batch_sizes,
            hidden_size=widths,
            learning_rate=[1e-3],
            weight_decay=[0.0],
            max_epochs=[1],
        ),
    )


def _batches(rows, batch_size):
    return [min(batch_size, rows - start) for start in range(0, rows, batch_size)]


@pytest.mark.parametrize(
    "accumulation_steps, microbatches",
    [(1, 1), (2, 2), (4, 4), (2, 1), (4, 3)],
)
def test_cached_accumulated_gradients_match_full_batch(
    monkeypatch, accumulation_steps, microbatches
):
    batch_size = 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        prepared = _prepared(train_rows=batch_size * microbatches)
    accelerator = Accelerator(
        cpu=True, mixed_precision="no", gradient_accumulation_steps=accumulation_steps
    )
    trainer = TorchProbeTrainer(
        TuneConfig(
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            early_stopping_patience=None,
            save_best_model=False,
        )
    )
    dataset = prepared.train_dataloader.dataset
    model = prepared.model
    model.train()
    model(
        inputs_embeds=dataset.embeddings,
        labels=torch.as_tensor(dataset.labels),
    ).loss.backward()
    expected = torch.cat(
        [
            parameter.grad.detach().clone().flatten()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )
    # Partial windows retain the configured divisor rather than renormalizing
    # by the number of microbatches actually present.
    expected *= microbatches / accumulation_steps
    assert expected.norm() > 0
    model.zero_grad(set_to_none=True)
    observed = []
    clip_grad_norm = accelerator.clip_grad_norm_

    def record_gradients(parameters, max_norm, **kwargs):
        parameters = list(parameters)
        observed.append(
            torch.cat(
                [
                    parameter.grad.detach().clone().flatten()
                    for parameter in parameters
                    if parameter.grad is not None
                ]
            )
        )
        return clip_grad_norm(parameters, max_norm, **kwargs)

    monkeypatch.setattr(accelerator, "clip_grad_norm_", record_gradients)

    trainer.train(
        MagicMock(task_type="sequence_classification", num_labels=2),
        prepared,
        accelerator,
        {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 1},
    )

    assert len(observed) == 1
    torch.testing.assert_close(
        observed[0],
        expected,
        msg=(
            f"Accumulation {accumulation_steps}, {microbatches} microbatches: "
            f"cached/reference gradient "
            f"norm ratio = {(observed[0].norm() / expected.norm()).item():.6f}"
        ),
    )


@pytest.mark.parametrize("batch_sizes", [[], [0], [-1], [8, 0]])
def test_batch_search_rejects_empty_or_nonpositive_batches(batch_sizes):
    with pytest.raises(ValidationError, match="batch_size"):
        HyperparameterSearchSpaceConfig(batch_size=batch_sizes)


@pytest.mark.parametrize("pattern", ["grid", "random"])
def test_batch_and_width_candidates_cover_the_joint_space(pattern):
    trainer = TorchProbeTrainer(
        TuneConfig(
            hyperparameter_search=_search(
                pattern, batch_sizes=_BATCH_SIZES, widths=[3, 5]
            )
        )
    )

    candidates = generate_candidates(
        trainer.config.hyperparameter_search.search_space,
        pattern,
        n_trials=10,
        seed=42,
    )

    assert len(candidates) == 10
    assert {(item["hidden_size"], item["batch_size"]) for item in candidates} == {
        (width, batch) for width in [3, 5] for batch in _BATCH_SIZES
    }
    assert candidates == generate_candidates(
        trainer.config.hyperparameter_search.search_space,
        pattern,
        n_trials=10,
        seed=42,
    )


@pytest.mark.parametrize("pattern", ["random", "bohb"])
def test_limited_search_covers_every_batch_size_and_width(
    monkeypatch, tmp_path, pattern
):
    if pattern == "bohb":
        pytest.importorskip("optuna")
    trainer = TorchProbeTrainer(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=4),
            hyperparameter_search=_search(
                pattern, batch_sizes=_BATCH_SIZES, widths=[3, 5, 7, 9], n_trials=5
            ),
            save_best_model=False,
        )
    )
    candidates = []

    def train(workspace, artifacts, accelerator, hparams, **kwargs):
        candidates.append(dict(hparams))
        return float(hparams["batch_size"])

    monkeypatch.setattr(trainer, "train", train)

    fit_probe_protocol(
        trainer,
        _make_workspace(tmp_path, "sequence_classification"),
        _prepared(train_rows=4, val_rows=4),
        accelerator=Accelerator(cpu=True),
        seed=42,
    )

    assert len(candidates) == 5
    assert {item["batch_size"] for item in candidates} == set(_BATCH_SIZES)
    assert {item["hidden_size"] for item in candidates[:4]} == {3, 5, 7, 9}


@pytest.mark.parametrize("batch_size", _BATCH_SIZES)
def test_trial_batch_size_controls_training_validation_and_scheduler(
    monkeypatch, batch_size
):
    prepared = _prepared()
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=3)
    trainer = TorchProbeTrainer(
        TuneConfig(
            batch_size=3,
            accumulation_steps=3,
            early_stopping_patience=5,
            save_best_model=False,
        )
    )
    optimizer_steps = []
    minibatches = {"train": [], "validation": []}
    forward = prepared.model.forward

    def record_forward(*args, **kwargs):
        phase = "train" if torch.is_grad_enabled() else "validation"
        minibatches[phase].append(kwargs["inputs_embeds"].shape[0])
        return forward(*args, **kwargs)

    def track_optimizer(*args, **kwargs):
        optimizer = build_optimizer(*args, **kwargs)
        optimizer.register_step_post_hook(lambda *args: optimizer_steps.append(None))
        return optimizer

    scheduler = MagicMock(wraps=build_scheduler)
    monkeypatch.setattr(prepared.model, "forward", record_forward)
    monkeypatch.setattr(
        "modules.evaluate.src.training.torch_trainer.build_optimizer", track_optimizer
    )
    monkeypatch.setattr(
        "modules.evaluate.src.training.torch_trainer.build_scheduler", scheduler
    )
    hparams = {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 2,
        "batch_size": batch_size,
    }

    score = trainer.train(
        MagicMock(task_type="sequence_classification", num_labels=2),
        prepared,
        accelerator,
        hparams,
    )

    assert score is not None and math.isfinite(score)
    assert minibatches["train"] == _batches(131, batch_size) * 2
    assert minibatches["validation"] == _batches(137, batch_size) * 2
    expected_steps = math.ceil(math.ceil(131 / batch_size) / 3) * 2
    assert len(optimizer_steps) == expected_steps
    assert scheduler.call_args.kwargs["num_training_steps"] == expected_steps


@pytest.mark.parametrize(
    "batch_size, train_batches, validation_batches",
    [
        (None, [6, 5], [5, 5, 3]),
        (3, [3, 3, 3, 2], [3, 3, 3, 3, 1]),
    ],
)
def test_fixed_batch_monitor_keeps_validation_loader_inheritance(
    monkeypatch, batch_size, train_batches, validation_batches
):
    prepared = _prepared(train_rows=11, val_rows=13)
    prepared.val_dataloader = DataLoader(prepared.val_dataloader.dataset, batch_size=5)
    trainer = TorchProbeTrainer(
        TuneConfig(batch_size=batch_size, save_best_model=False)
    )
    minibatches = {"train": [], "validation": []}
    forward = prepared.model.forward

    def record_forward(*args, **kwargs):
        phase = "train" if torch.is_grad_enabled() else "validation"
        minibatches[phase].append(kwargs["inputs_embeds"].shape[0])
        return forward(*args, **kwargs)

    monkeypatch.setattr(prepared.model, "forward", record_forward)

    trainer.train(
        MagicMock(task_type="sequence_classification", num_labels=2),
        prepared,
        Accelerator(cpu=True),
        {"learning_rate": 1e-3, "weight_decay": 0.0, "max_epochs": 1},
    )

    assert minibatches["train"] == train_batches
    assert minibatches["validation"] == validation_batches


@pytest.mark.parametrize("pattern", ["grid", "random", "bohb"])
def test_batch_search_preserves_winner_for_fallback_validation_variance_and_saving(
    monkeypatch, tmp_path, pattern
):
    if pattern == "bohb":
        pytest.importorskip("optuna")
    trainer = TorchProbeTrainer(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=4),
            batch_size=3,
            variance_seeds=[1, 2],
            hyperparameter_search=_search(
                pattern, batch_sizes=[8, 16], widths=[3, 5], n_trials=4
            ),
        )
    )
    prepared = _prepared(train_rows=17, val_rows=19)
    workspace = _make_workspace(tmp_path, "sequence_classification")
    trials = []
    validations = []

    def train(workspace, artifacts, accelerator, hparams, **kwargs):
        with torch.no_grad():
            for parameter in artifacts.model.parameters():
                parameter.fill_((len(trials) + 1) / 100)
        trials.append(
            {
                "hparams": dict(hparams),
                "state": {
                    name: value.clone()
                    for name, value in artifacts.model.state_dict().items()
                },
            }
        )
        # No epoch-level score: HPO must use its fallback validation path.

    def validate(
        workspace, artifacts, accelerator, metric_name, *, batch_size=None, **kwargs
    ):
        current = trials[-1]["hparams"]
        assert batch_size == current["batch_size"]
        validations.append(batch_size)
        return current["batch_size"] + current["hidden_size"] / 100

    monkeypatch.setattr(trainer, "train", train)
    monkeypatch.setattr(trainer, "evaluate_validation", validate)

    result = fit_probe_protocol(
        trainer, workspace, prepared, seed=42, accelerator=Accelerator(cpu=True)
    )

    search_trials = trials[:4]
    best = max(
        search_trials,
        key=lambda trial: (
            trial["hparams"]["batch_size"] + trial["hparams"]["hidden_size"] / 100
        ),
    )
    assert len(trials) == 6
    assert len(validations) == 6
    assert all(trial["hparams"]["batch_size"] in [8, 16] for trial in search_trials)
    if pattern != "bohb":
        assert {
            (trial["hparams"]["hidden_size"], trial["hparams"]["batch_size"])
            for trial in search_trials
        } == {(width, batch) for width in [3, 5] for batch in [8, 16]}
    assert all(trial["hparams"] == best["hparams"] for trial in trials[4:])
    for name, value in best["state"].items():
        torch.testing.assert_close(result.model.state_dict()[name], value)
    metadata = json.loads((workspace.model_path / "probe_head_config.json").read_text())
    assert (
        metadata["best_hyperparameters"]["batch_size"] == best["hparams"]["batch_size"]
    )
    assert (
        metadata["best_hyperparameters"]["hidden_size"]
        == best["hparams"]["hidden_size"]
    )


@pytest.mark.parametrize(
    "unsupported", ["torch_linear", "uncached", "finetune", "layerwise"]
)
def test_batch_search_rejects_unsupported_probe_paths(tmp_path, unsupported):
    prepared = _prepared(train_rows=4, val_rows=4)
    if unsupported == "uncached":
        prepared.train_dataloader = DataLoader([{"labels": 0}], batch_size=1)
    elif unsupported == "layerwise":
        prepared.model = MagicMock(spec=LayerwiseProbeModel)
    trainer = TorchProbeTrainer(
        TuneConfig(
            head=HeadConfig(
                head_type="torch_linear" if unsupported == "torch_linear" else "mlp"
            ),
            hyperparameter_search=_search("grid", batch_sizes=[8, 16]),
        ),
        train_trunk=unsupported == "finetune",
    )

    with pytest.raises(
        ValueError, match="batch_size search requires a cached MLP probe"
    ):
        fit_probe_protocol(
            trainer,
            _make_workspace(tmp_path, "sequence_classification"),
            prepared,
            accelerator=Accelerator(cpu=True),
        )


def test_disabled_batch_search_keeps_fixed_training_configuration(
    monkeypatch, tmp_path
):
    search = _search("grid", batch_sizes=_BATCH_SIZES)
    search.enabled = False
    trainer = TorchProbeTrainer(
        TuneConfig(
            head=HeadConfig(head_type="mlp", hidden_size=4),
            batch_size=32,
            hyperparameter_search=search,
            save_best_model=False,
        )
    )
    train = MagicMock()
    monkeypatch.setattr(trainer, "train", train)

    fit_probe_protocol(
        trainer,
        _make_workspace(tmp_path, "sequence_classification"),
        _prepared(train_rows=4, val_rows=4),
        accelerator=Accelerator(cpu=True),
    )

    assert "batch_size" not in train.call_args.args[3]
    assert trainer.config.batch_size == 32


def test_default_mlp_enables_requested_batch_size_choices():
    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    config = load_and_parse(
        path=[
            root / "config.yaml",
            root / "tasks/sequence_classification.yaml",
            root / "datasets/localization_prediction.yaml",
        ],
        model_cls=EvaluationConfig,
    )

    search = config.steps.tune.hyperparameter_search
    assert search.enabled
    assert search.search_space.batch_size == [*_BATCH_SIZES, 256]
