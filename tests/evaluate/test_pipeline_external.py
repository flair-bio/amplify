"""Orchestration of external test sets in run_pipeline, with steps stubbed."""

from __future__ import annotations

from types import SimpleNamespace

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners import pipeline


def _config(mode: str) -> EvaluationConfig:
    return EvaluationConfig.model_validate(
        {
            "workspace": {
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
    prepared = SimpleNamespace(model=object(), test_dataloader="main")

    class Prepare:
        def __init__(self, config):
            pass

        def run(self, workspace, **kwargs):
            return prepared

        def prepare_external_test(self, workspace, base, external, **kwargs):
            return SimpleNamespace(model=base.model, test_dataloader=external.name)

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
            return SimpleNamespace()

    class Score:
        def __init__(self, config):
            pass

        def run(self, workspace, **kwargs):
            return SimpleNamespace(scores={}, summary_path=None)

    for name, cls in (
        ("PrepareStep", Prepare),
        ("TuneStep", Tune),
        ("PredictStep", Predict),
        ("ScoreStep", Score),
    ):
        monkeypatch.setattr(pipeline, name, cls)


class _Accelerator:
    is_main_process = True

    def prepare(self, model):
        return model

    def wait_for_everyone(self):
        pass


def test_external_set_reuses_fitted_model_without_retuning(monkeypatch, tmp_path):
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
