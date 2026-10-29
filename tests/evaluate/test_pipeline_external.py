"""Orchestration of external test sets in run_pipeline, with steps stubbed."""

from __future__ import annotations

import json
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from modules.evaluate.src.contracts.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners import pipeline
from modules.evaluate.src.contracts.artifacts import EvaluationMetadata, PredictOutput, ScoreOutput


def _config(mode: str) -> EvaluationConfig:
    return EvaluationConfig.model_validate(
        {
            "workspace": {
                "run_id": "test",
                "dataset_name": "loc",
                "dataset_repo_id": "org/loc",
                "model_name": "m",
                "task_type": "sequence_classification",
                "target_type": "categorical",
                "label_format": "scalar",
                "num_labels": 2,
            },
            "steps": {"mode": mode},
            "external_test_sets": [
                {"name": "sethard", "dataset_repo_id": "org/sethard"}
            ],
        }
    )


def _stub_steps(monkeypatch, calls):
    prepared = SimpleNamespace(
        contact=None, embedding=None, layerwise=None,
        model=object(), test_dataloader="main",
    )

    class Prepare:
        def __init__(self, config):
            pass

        def run(self, workspace, **kwargs):
            return prepared

        def prepare_external_test(self, workspace, base, external, **kwargs):
            return SimpleNamespace(contact=None, model=base.model, test_dataloader=external.name)

    class Tune:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, workspace, artifacts, **kwargs):
            calls.append(("tune", id(artifacts.model)))
            return artifacts

    class Predict:
        def __init__(self, config):
            pass

        def run(self, workspace, artifacts, **kwargs):
            calls.append(
                (
                    workspace.dataset_repo_id,
                    artifacts.test_dataloader,
                    id(artifacts.model),
                    workspace.variant,
                )
            )
            return PredictOutput(
                metadata=EvaluationMetadata(
                    model_name=workspace.model_name,
                    dataset_name=workspace.dataset_name,
                    task_type=workspace.task_type,
                    variant=workspace.variant,
                ),
                predictions=pl.DataFrame(),
            )

    class Score:
        def __init__(self, config):
            pass

        def run(self, workspace, **kwargs):
            return ScoreOutput(scores={})

    for name, cls in (
        ("PrepareStep", Prepare),
        ("TuneStep", Tune),
        ("PredictStep", Predict),
        ("ScoreStep", Score),
    ):
        monkeypatch.setattr(pipeline, name, cls)


class _Accelerator:
    is_main_process = True
    device = torch.device("cpu")

    def prepare(self, model):
        return model

    def wait_for_everyone(self):
        pass


def test_external_set_reuses_fitted_model_without_retuning(
    monkeypatch, tmp_path, caplog
):
    caplog.set_level("INFO")
    calls: list = []
    _stub_steps(monkeypatch, calls)
    config = _config("tune_probe")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)

    result = pipeline.run_pipeline(config, workspace, _Accelerator())

    assert calls[0][0] == "tune"
    assert calls[1][:2] == ("org/loc", "main")
    assert calls[2][:2] == ("org/sethard", "sethard")
    assert calls[1][2] == calls[2][2] == calls[0][1]
    assert calls[1][3] == calls[2][3] == "mlp-frozen_trunk"
    assert set(result.external) == {"sethard"}
    assert result.evaluations[0].prediction is result.predictions[None]
    assert result.evaluations[0].score is result.scores[None]
    assert "Preparing evaluation inputs" in (workspace.run_root / "log").read_text()
    assert not list(tmp_path.rglob("logs"))


@pytest.mark.parametrize("mode", ["tune_probe", "evaluate_as_is"])
def test_controls_step_runs_once_before_prediction(monkeypatch, tmp_path, mode):
    calls = []
    _stub_steps(monkeypatch, calls)

    class Controls:
        def __init__(self, config, prepare_config):
            pass

        def run(self, workspace, prepared, *, fitted, controls, **kwargs):
            assert fitted == (mode == "tune_probe")
            assert controls == ["random_trunk"]
            calls.append(("controls", id(prepared.model)))
            return prepared

    monkeypatch.setattr(pipeline, "ControlsStep", Controls)
    config = _config(mode)
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    pipeline.run_pipeline(config, workspace, _Accelerator())
    stages = [call[0] for call in calls]
    assert stages == (
        ["tune", "controls", "org/loc", "org/sethard"]
        if mode == "tune_probe" else ["controls", "org/loc"]
    )
    assert len({call[1] for call in calls if call[0] in {"tune", "controls"}}) == 1


def test_external_sets_skipped_without_fitted_probe(monkeypatch, tmp_path):
    calls: list = []
    _stub_steps(monkeypatch, calls)
    config = _config("evaluate_as_is")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)

    result = pipeline.run_pipeline(config, workspace, _Accelerator())

    assert len(calls) == 1
    assert calls[0][:2] == ("org/loc", "main")
    assert result.external == {}


def test_rerun_is_rejected_before_preparing_or_tuning(monkeypatch, tmp_path):
    calls = []
    _stub_steps(monkeypatch, calls)
    config = _config("tune_probe")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    pipeline.run_pipeline(config, workspace, _Accelerator())
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["evaluation"]["config"]["workspace"]["run_id"] == "test"
    calls.clear()
    with pytest.raises(FileExistsError, match="--overwrite"):
        pipeline.run_pipeline(config, workspace, _Accelerator())
    assert calls == []
    pipeline.run_pipeline(config, workspace, _Accelerator(), overwrite=True)
    assert len(calls) == 3


def test_external_preflight_failure_preserves_main_outputs(monkeypatch, tmp_path):
    calls = []
    _stub_steps(monkeypatch, calls)
    config = _config("tune_probe")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    pipeline.run_pipeline(config, workspace, _Accelerator())
    before = workspace.run_manifest_path.read_bytes()
    calls.clear()
    # Reusing an external label for another source is forbidden even with overwrite.
    config.external_test_sets[0].dataset_repo_id = "org/different"
    with pytest.raises(ValueError, match="different inputs"):
        pipeline.run_pipeline(config, workspace, _Accelerator(), overwrite=True)
    assert workspace.run_manifest_path.read_bytes() == before
    assert calls == []


def test_main_rank_preflight_errors_are_broadcast(monkeypatch, tmp_path):
    config = _config("tune_probe")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    workspace.run_root.mkdir(parents=True)
    workspace.run_manifest_path.write_text("{}")
    messages = []

    def broadcast(values, **kwargs):
        messages.extend(values)
        return values

    monkeypatch.setattr(pipeline, "broadcast_object_list", broadcast)
    accelerator = SimpleNamespace(is_main_process=True, num_processes=2)
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(config, workspace, accelerator)
    assert "--overwrite" in messages[0]


def test_worker_rank_receives_preflight_errors(monkeypatch, tmp_path):
    config = _config("tune_probe")
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    monkeypatch.setattr(
        pipeline,
        "broadcast_object_list",
        lambda *args, **kwargs: ["Output target already exists"],
    )
    accelerator = SimpleNamespace(is_main_process=False, num_processes=2)
    with pytest.raises(RuntimeError, match="already exists"):
        pipeline.run_pipeline(config, workspace, accelerator)
