import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from modules.post_train.src.contracts.config import PostTrainingConfig, WandbConfig
from modules.post_train.src.tracking.progress import track_run
from modules.post_train.src.trainers.latent_alignment import capture_rng


@pytest.fixture
def reporting_config(tmp_path):
    return PostTrainingConfig(
        student={"checkpoint": "/unused/student"},
        dataset={"path": "/unused/data"},
        output_dir=tmp_path,
    )


@pytest.mark.parametrize("enabled,mode", [(False, "offline"), (True, "disabled")])
def test_progress_preserves_jsonl_without_importing_wandb(
    reporting_config, tmp_path, monkeypatch, capsys, enabled, mode,
):
    reporting_config.wandb = WandbConfig(enabled=enabled, mode=mode)
    monkeypatch.setitem(sys.modules, "wandb", None)
    rows = [
        {"step": 0, "validation": {"mlm": 2.0}},
        {"step": 1, "mlm": 1.5, "latent": 4.0, "total": 3.5},
    ]
    with track_run(reporting_config, tmp_path, {}, 1, torch.device("cpu")) as logger:
        logger.phase("Validating (initial)")
        for row in rows:
            logger.log(row)
    assert logger.stream.closed
    assert [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()] == rows
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Post-training" in captured.err
    assert "val_mlm=2.0000" in captured.err
    assert '{"step":' not in captured.err


def consume_rng(*args, **kwargs):
    random.random()
    np.random.random()
    torch.rand(1)


def test_wandb_records_namespaced_metrics_and_does_not_change_training_rng(
    reporting_config, tmp_path, monkeypatch,
):
    tracker = Mock()
    tracker.log.side_effect = consume_rng

    def initialize(**kwargs):
        consume_rng()
        return tracker

    sdk = SimpleNamespace(init=Mock(side_effect=initialize), Settings=Mock())
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    reporting_config.wandb = WandbConfig(enabled=True, run_name="cpu-fixture")
    reporting_config.training.progress_bar = False
    device = torch.device("cpu")
    before = capture_rng(device)
    resolved = {"training": {"device": "cpu"}}
    with track_run(reporting_config, tmp_path, resolved, 1, device) as logger:
        logger.log({
            "step": 0, "validation": {"mlm": 2.0, "sources": {"atlas": 2.0}},
        })
        logger.log({
            "step": 1, "total": 3.0,
            "gradient_norms_before_clipping": {"encoder": 0.5},
        })
    after = capture_rng(device)
    assert before[0] == after[0]
    np.testing.assert_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert sdk.init.call_args.kwargs["mode"] == "offline"
    assert sdk.init.call_args.kwargs["config"] == resolved
    sdk.Settings.assert_called_once_with(disable_code=True, disable_git=True)
    assert tracker.log.call_args_list[0].kwargs == {"step": 0}
    assert tracker.log.call_args_list[0].args[0] == {
        "validation/mlm": 2.0, "validation/sources/atlas": 2.0,
    }
    assert tracker.log.call_args_list[1].args[0] == {
        "train/total": 3.0, "train/gradient_norms_before_clipping/encoder": 0.5,
    }
    tracker.finish.assert_called_once_with(exit_code=0)


def test_wandb_is_finished_as_failed_and_local_logs_closed_on_error(
    reporting_config, tmp_path, monkeypatch,
):
    tracker = Mock()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=Mock(return_value=tracker), Settings=Mock(),
    ))
    reporting_config.wandb.enabled = True
    reporting_config.training.progress_bar = False
    with pytest.raises(RuntimeError, match="training failed"):
        with track_run(reporting_config, tmp_path, {}, 1, torch.device("cpu")) as logger:
            raise RuntimeError("training failed")
    assert logger.stream.closed
    tracker.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("options", [{"mode": "bad"}, {"project": ""}])
def test_invalid_tracking_config(options):
    with pytest.raises(ValidationError):
        WandbConfig(**options)
