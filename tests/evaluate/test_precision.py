"""Evaluation entrypoints apply TF32 defaults and CLI overrides before model work."""

from __future__ import annotations

import importlib
import json
import sys
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
    "entrypoint_name",
    ["run_evaluate", "run_proteingym_zero_shot", "run_proteingym_supervised"],
)
@pytest.mark.parametrize(
    "override", [None, "true", "false"], ids=["default", "on", "off"]
)
def test_cli_applies_tf32_before_model_work(
    entrypoint_name, override, monkeypatch, tmp_path, restore_precision_settings
):
    entrypoint = importlib.import_module(f"modules.evaluate.src.{entrypoint_name}")
    expected = override != "false"
    # Starting from the opposite state checks that both enabling and disabling
    # affect this run, rather than inheriting a previous run's global settings.
    torch.set_float32_matmul_precision("highest" if expected else "high")
    torch.backends.cuda.matmul.allow_tf32 = not expected
    torch.backends.cudnn.allow_tf32 = not expected

    if entrypoint_name == "run_evaluate":
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
            "model_id": "offline_model",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "out"),
            "device": "cpu",
        }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(payload))
    argv = [entrypoint_name, str(config_path)]
    if override is not None:
        argv.append(f"tf32={override}")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(entrypoint, "seed_everything", lambda seed: None)

    class ModelWorkReached(Exception):
        pass

    def check_settings(*args, **kwargs):
        assert torch.get_float32_matmul_precision() == (
            "high" if expected else "highest"
        )
        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
        if entrypoint_name == "run_evaluate":
            assert args[0].tf32 is expected
        raise ModelWorkReached

    if entrypoint_name == "run_evaluate":
        monkeypatch.setattr(
            entrypoint,
            "Accelerator",
            lambda **kwargs: SimpleNamespace(is_main_process=True),
        )
        monkeypatch.setattr(
            entrypoint, "get_evaluation_workspace", lambda config, **kwargs: config
        )
        monkeypatch.setattr(entrypoint, "run_evaluation", check_settings)
    else:
        monkeypatch.setattr(
            entrypoint.AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: object(),
        )
        model_class = (
            entrypoint.AutoModelForMaskedLM
            if entrypoint_name == "run_proteingym_zero_shot"
            else entrypoint.AutoModel
        )
        monkeypatch.setattr(model_class, "from_pretrained", check_settings)

    # Stop at model work so this exercises the real CLI/configuration path
    # without loading datasets, downloading checkpoints or requiring a GPU.
    with pytest.raises(ModelWorkReached):
        if entrypoint_name == "run_evaluate":
            # The suite CLI requires --suite; the single-run body is what applies
            # TF32 and reaches run_evaluation, so exercise it with the same overrides.
            entrypoint._run_model_evaluation(
                [str(config_path)],
                [] if override is None else [f"tf32={override}"],
                False,
            )
        else:
            entrypoint.main()
