"""Engine factory: which configs run on the generic runner and how ambiguity fails."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.evaluate.src.tasks.contact import ContactTask
from modules.evaluate.src.tasks.engines import EngineSpec, resolve_engine
from modules.evaluate.src.tasks.frozen_probe import FrozenProbeTask
from modules.evaluate.src.tasks.masked_lm import MaskedLMTask


def _config(
    task_type="sequence_regression",
    mode="tune_probe",
    layerwise=False,
    head_type="mlp",
):
    return SimpleNamespace(
        workspace=SimpleNamespace(task_type=task_type),
        steps=SimpleNamespace(
            mode=mode,
            layerwise=SimpleNamespace(enabled=layerwise),
            tune=SimpleNamespace(head=SimpleNamespace(head_type=head_type)),
        ),
    )


@pytest.mark.parametrize("mode", ["tune_probe", "finetune", "evaluate_as_is"])
@pytest.mark.parametrize(
    "task_type",
    [
        "sequence_regression",
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    ],
)
def test_supervised_sequence_jobs_resolve_to_frozen_probe(task_type, mode) -> None:
    task = resolve_engine(_config(task_type=task_type, mode=mode))
    assert isinstance(task, FrozenProbeTask)


def test_unclaimed_contact_head_has_no_engine() -> None:
    config = _config(task_type="contact_prediction", head_type="unclaimed_head")
    assert resolve_engine(config) is None


@pytest.mark.parametrize(
    ("head_type", "predictor_name"),
    [
        ("bilinear512", "contact_embedding"),
        ("attention_l1", "contact_attention"),
        ("categorical_jacobian", "contact_jacobian"),
    ],
)
def test_contact_heads_resolve_to_contact_task(head_type, predictor_name) -> None:
    task = resolve_engine(_config(task_type="contact_prediction", head_type=head_type))
    assert isinstance(task, ContactTask)
    assert task.predictor_type.name == predictor_name


@pytest.mark.parametrize("task_type", ["calibration", "pseudo_perplexity"])
def test_zero_shot_mlm_jobs_resolve_to_masked_lm(task_type) -> None:
    task = resolve_engine(_config(task_type=task_type, mode="evaluate_as_is"))
    assert isinstance(task, MaskedLMTask)


def test_layerwise_zero_shot_mlm_jobs_are_unclaimed() -> None:
    config = _config(task_type="calibration", mode="evaluate_as_is", layerwise=True)
    assert resolve_engine(config) is None


def test_layerwise_sequence_jobs_resolve_to_frozen_probe() -> None:
    assert isinstance(resolve_engine(_config(layerwise=True)), FrozenProbeTask)


def test_layerwise_token_jobs_resolve_to_frozen_probe() -> None:
    config = _config(task_type="token_classification", layerwise=True)
    assert isinstance(resolve_engine(config), FrozenProbeTask)


def test_ambiguous_engines_are_rejected() -> None:
    spec = EngineSpec(name="a", supports=lambda c: True, build=lambda c: None)
    other = EngineSpec(name="b", supports=lambda c: True, build=lambda c: None)
    with pytest.raises(ValueError, match="a, b"):
        resolve_engine(_config(), engines=(spec, other))


_GRID_TASKS = {
    "sequence_regression": ("continuous", "scalar", None),
    "sequence_classification": ("categorical", "scalar", 2),
    "token_classification": ("categorical", "per_token", 2),
    "sequence_multilabel_classification": ("categorical", "multilabel", 2),
    "calibration": ("categorical", "consensus_profile", None),
    "pseudo_perplexity": ("continuous", "scalar", None),
}


def _load_grid_config(tmp_path, task_type, mode, layerwise):
    """Load a config through the real validator; raises when validation rejects it."""
    import json

    from modules.evaluate.src.utils.config_loader import load_evaluation_config

    target_type, label_format, num_labels = _GRID_TASKS[task_type]
    steps: dict = {
        "mode": mode,
        "tune": {"hyperparameter_search": {"enabled": False}, "max_epochs": 1},
    }
    if mode in {"finetune", "evaluate_as_is"}:
        # Config validation requires the uncached path for these modes.
        steps["prepare"] = {"embedding_cache": {"enabled": False}}
    if layerwise:
        steps["layerwise"] = {"enabled": True, "layers": [0]}
        steps["tune"]["head"] = {"head_type": "torch_linear"}
    payload = {
        "seed": 1,
        "workspace": {
            "run_id": "grid",
            "model_name": "offline_model",
            "model_repo_id": "flair-bio/offline-model",
            "dataset_name": "offline_dataset",
            "dataset_repo_id": "org/offline_dataset",
            "task_type": task_type,
            "target_type": target_type,
            "label_format": label_format,
            "num_labels": num_labels,
            "base_path": str(tmp_path / "eval"),
        },
        "steps": steps,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return load_evaluation_config(path=[str(path)])


_SUPERVISED = ("sequence_regression", "sequence_classification", "token_classification")
_SUPERVISED_ALL = (*_SUPERVISED, "sequence_multilabel_classification")

# Every (task, mode, layerwise) cell of the grid, split by validation outcome.
# Accepted cells must resolve to their expected engine type.
_ACCEPTED_GRID = [
    *[
        (t, m, False)
        for t in _SUPERVISED_ALL
        for m in ("tune_probe", "finetune", "evaluate_as_is")
    ],
    *[(t, "tune_probe", True) for t in _SUPERVISED_ALL],
    ("calibration", "evaluate_as_is", False),
    ("pseudo_perplexity", "evaluate_as_is", False),
]
# Rejected cells: layerwise is not offered for finetune/evaluate_as_is supervised
# jobs, zero-shot MLM jobs do not train a probe, and layerwise zero-shot is unsupported.
_REJECTED_GRID = [
    *[(t, m, True) for t in _SUPERVISED_ALL for m in ("finetune", "evaluate_as_is")],
    ("calibration", "tune_probe", False),
    ("calibration", "tune_probe", True),
    ("calibration", "finetune", False),
    ("calibration", "finetune", True),
    ("calibration", "evaluate_as_is", True),
    ("pseudo_perplexity", "tune_probe", False),
    ("pseudo_perplexity", "tune_probe", True),
    ("pseudo_perplexity", "finetune", False),
    ("pseudo_perplexity", "finetune", True),
    ("pseudo_perplexity", "evaluate_as_is", True),
]


def test_grid_partitions_every_cell() -> None:
    """The accepted and rejected tables must cover all 36 grid cells exactly once."""
    accepted, rejected = set(_ACCEPTED_GRID), set(_REJECTED_GRID)
    assert len(_ACCEPTED_GRID) == len(accepted) == 18
    assert len(_REJECTED_GRID) == len(rejected) == 18
    assert accepted.isdisjoint(rejected)
    all_cells = {
        (t, m, lw)
        for t in _GRID_TASKS
        for m in ("tune_probe", "finetune", "evaluate_as_is")
        for lw in (False, True)
    }
    assert accepted | rejected == all_cells


@pytest.mark.parametrize(("task_type", "mode", "layerwise"), _ACCEPTED_GRID)
def test_accepted_grid_cells_resolve_an_engine(tmp_path, task_type, mode, layerwise):
    """Validation is the contract: an accepted config must always find an engine."""
    config = _load_grid_config(tmp_path, task_type, mode, layerwise)
    engine = resolve_engine(config)
    expected = (
        MaskedLMTask
        if task_type in {"calibration", "pseudo_perplexity"}
        else FrozenProbeTask
    )
    assert isinstance(engine, expected)


@pytest.mark.parametrize(("task_type", "mode", "layerwise"), _REJECTED_GRID)
def test_rejected_grid_cells_fail_validation(tmp_path, task_type, mode, layerwise):
    """Rejected cells must fail at config validation, never reach engine resolution."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _load_grid_config(tmp_path, task_type, mode, layerwise)


@pytest.mark.parametrize(
    ("task_type", "mode", "layerwise"),
    [
        ("sequence_regression", "tune_probe", False),
        ("sequence_classification", "tune_probe", True),
        ("token_classification", "finetune", False),
        ("calibration", "evaluate_as_is", False),
        ("pseudo_perplexity", "evaluate_as_is", False),
    ],
)
def test_known_valid_grid_configs_are_not_vacuously_skipped(
    tmp_path, task_type, mode, layerwise
) -> None:
    """Guards the grid above: these configurations must load, or it proves nothing."""
    config = _load_grid_config(tmp_path, task_type, mode, layerwise)
    assert resolve_engine(config) is not None
