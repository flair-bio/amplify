"""Evaluation entrypoints apply TF32 defaults and CLI overrides before model work."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def restore_precision_settings():
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(
    "workflow", ["evaluation", "proteingym_zero_shot", "proteingym_supervised"]
)
@pytest.mark.parametrize(
    "override", [None, "true", "false"], ids=["default", "on", "off"]
)
def test_cli_applies_tf32_before_model_work(
    workflow, override, monkeypatch, tmp_path, restore_precision_settings
):
    from modules.evaluate.src import run_evaluate
    from modules.evaluate.src.runners import runtime

    expected = override != "false"
    torch.set_float32_matmul_precision("highest" if expected else "high")
    torch.backends.cuda.matmul.allow_tf32 = not expected
    torch.backends.cudnn.allow_tf32 = not expected
    if workflow == "evaluation":
        payload = {
            "workspace": {
                "run_id": "test",
                "model_name": "offline_model",
                "dataset_name": "offline_dataset",
                "task_type": "sequence_classification",
                "target_type": "categorical",
                "label_format": "scalar",
                "num_labels": 2,
            }
        }
    else:
        payload = {
            "workflow": workflow,
            "model_id": "offline_model",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "device": "cpu",
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(payload))
    argv = [str(config_path)]
    if override is not None:
        argv.append(f"tf32={override}")
    monkeypatch.setattr(runtime, "seed_everything", lambda seed: None)
    monkeypatch.setattr(
        runtime,
        "Accelerator",
        lambda **kwargs: SimpleNamespace(
            is_main_process=True, device=torch.device("cpu")
        ),
    )

    class ModelWorkReached(Exception):
        pass

    def check_settings(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == (
            "high" if expected else "highest"
        )
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
        assert args[0].tf32 is expected
        assert args[0].workflow == workflow
        raise ModelWorkReached

    monkeypatch.setattr(run_evaluate, "dispatch_run", check_settings)
    with pytest.raises(ModelWorkReached):
        run_evaluate.main(argv)
