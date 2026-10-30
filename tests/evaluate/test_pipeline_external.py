"""External test-set orchestration of the generic runner.

The output reservation, rerun guard and external preflight run for real against a
temporary workspace. The task and predictor are the fakes from
test_evaluation_runner, so the lifecycle is asserted at the runner boundary.
"""

from __future__ import annotations

import json

import pytest

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.evaluation import reservation
from modules.evaluate.src.evaluation.runner import run_evaluation
from tests.evaluate.test_evaluation_runner import (
    FakePredictor,
    FakeTask,
    _Accelerator,
)


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


def _setup(tmp_path, mode: str):
    """Config and real output workspace rooted in a temporary directory."""
    config = _config(mode)
    config.workspace.base_path = str(tmp_path)
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
    return config, workspace


def test_external_set_reuses_fitted_model_without_retuning(tmp_path, caplog):
    caplog.set_level("INFO")
    config, workspace = _setup(tmp_path, "tune_probe")
    log: list = []

    result = run_evaluation(config, workspace, _Accelerator(), task=FakeTask(log))

    assert log.count(("fit",)) == 1
    assert log.index(("fit",)) < log.index(("predict", "test", None))
    assert log.index(("predict", "test", None)) < log.index(
        ("prepare_external", "sethard")
    )
    assert ("predict", "sethard", "sethard") in log
    assert set(result.external) == {"sethard"}
    assert result.external["sethard"].predictions[0].split == "sethard"
    external_workspace = result.external["sethard"].predictions[0].workspace
    assert external_workspace.dataset_repo_id == "org/sethard"
    assert external_workspace.variant == workspace.variant == "mlp-frozen_trunk"
    assert "Starting Prepare Step" in (workspace.run_root / "log").read_text()
    assert not list(tmp_path.rglob("logs"))


def test_external_sets_skipped_without_fitted_probe(tmp_path, caplog):
    caplog.set_level("WARNING")
    config, workspace = _setup(tmp_path, "evaluate_as_is")

    class NonTransferringPredictor(FakePredictor):
        transfers_to_external = False

    class EvaluateAsIsTask(FakeTask):
        def predictor(self, context):
            return NonTransferringPredictor(self.log)

    log: list = []
    result = run_evaluation(
        config, workspace, _Accelerator(), task=EvaluateAsIsTask(log)
    )

    assert ("prepare_external", "sethard") not in log
    assert [entry for entry in log if entry[0] == "predict"] == [
        ("predict", "test", None)
    ]
    assert result.external == {}
    assert "Skipping external_test_sets" in caplog.text
    assert "does not transfer" in caplog.text


def test_rerun_is_rejected_before_preparing_or_tuning(tmp_path):
    config, workspace = _setup(tmp_path, "tune_probe")
    log: list = []
    run_evaluation(config, workspace, _Accelerator(), task=FakeTask(log))
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["evaluation"]["config"]["workspace"]["run_id"] == "test"
    log.clear()
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_evaluation(config, workspace, _Accelerator(), task=FakeTask(log))
    assert log == []
    run_evaluation(
        config, workspace, _Accelerator(), task=FakeTask(log), overwrite=True
    )
    assert log.count(("fit",)) == 1
    assert ("prepare_external", "sethard") in log


def test_external_preflight_failure_preserves_main_outputs(tmp_path):
    config, workspace = _setup(tmp_path, "tune_probe")
    run_evaluation(config, workspace, _Accelerator(), task=FakeTask([]))
    before = workspace.run_manifest_path.read_bytes()
    log: list = []
    # Reusing an external label for another source is forbidden even with overwrite.
    config.external_test_sets[0].dataset_repo_id = "org/different"
    with pytest.raises(ValueError, match="different inputs"):
        run_evaluation(
            config, workspace, _Accelerator(), task=FakeTask(log), overwrite=True
        )
    assert workspace.run_manifest_path.read_bytes() == before
    assert log == []


def test_main_rank_preflight_errors_are_broadcast(monkeypatch, tmp_path):
    config, workspace = _setup(tmp_path, "tune_probe")
    workspace.run_root.mkdir(parents=True)
    workspace.run_manifest_path.write_text("{}")
    messages = []

    def broadcast(values, **kwargs):
        messages.extend(values)
        return values

    monkeypatch.setattr(reservation, "broadcast_object_list", broadcast)
    accelerator = _Accelerator(main=True, num_processes=2)
    with pytest.raises(FileExistsError):
        run_evaluation(config, workspace, accelerator, task=FakeTask([]))
    assert "--overwrite" in messages[0]


def test_worker_rank_receives_preflight_errors(monkeypatch, tmp_path):
    config, workspace = _setup(tmp_path, "tune_probe")
    monkeypatch.setattr(
        reservation,
        "broadcast_object_list",
        lambda *args, **kwargs: ["Output target already exists"],
    )
    accelerator = _Accelerator(main=False, num_processes=2)
    with pytest.raises(RuntimeError, match="already exists"):
        run_evaluation(config, workspace, accelerator, task=FakeTask([]))
