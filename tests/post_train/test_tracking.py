import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from modules.post_train.src.config import PostTrainingConfig, WandbConfig
from modules.post_train.src.utils.tracking import (
    log_wandb,
    wandb_metrics,
    wandb_run,
)
from modules.post_train.src.trainers.latent_alignment import capture_rng


@pytest.fixture
def reporting_config(tmp_path):
    return PostTrainingConfig(
        student={"checkpoint": "/unused/student"},
        dataset={"path": "/unused/data"},
        output_dir=tmp_path,
    )


def test_disabled_wandb_does_not_start_or_log(reporting_config, tmp_path, monkeypatch):
    reporting_config.wandb = WandbConfig(enabled=False)
    sdk = SimpleNamespace(run=None, init=Mock(), log=Mock())
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    with wandb_run(reporting_config, tmp_path, {}, torch.device("cpu")):
        log_wandb({"train/loss": 1.0}, step=1, device=torch.device("cpu"))
    sdk.init.assert_not_called()
    sdk.log.assert_not_called()


def consume_rng(*args, **kwargs):
    random.random()
    np.random.random()
    torch.rand(1)


def test_wandb_logs_namespaced_metrics_without_changing_training_rng(
    reporting_config,
    tmp_path,
    monkeypatch,
):
    tracker = Mock()
    tracker.log.side_effect = consume_rng

    def initialize(**kwargs):
        consume_rng()
        return tracker

    sdk = SimpleNamespace(
        init=Mock(side_effect=initialize),
        Settings=Mock(),
        run=tracker,
        log=tracker.log,
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    reporting_config.wandb = WandbConfig(enabled=True, run_name="cpu-fixture")
    device = torch.device("cpu")
    before = capture_rng(device)
    resolved = {"training": {"device": "cpu"}}
    with wandb_run(reporting_config, tmp_path, resolved, device):
        rows = [
            {"step": 0, "validation": {"mlm": 2.0, "sources": {"atlas": 2.0}}},
            {
                "step": 1,
                "total": 3.0,
                "gradient_norms_before_clipping": {"encoder": 0.5},
            },
        ]
        for row in rows:
            log_wandb(wandb_metrics(row), step=row["step"], device=device)
    after = capture_rng(device)
    assert before[0] == after[0]
    np.testing.assert_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert sdk.init.call_args.kwargs["config"] == resolved
    assert "mode" not in sdk.init.call_args.kwargs
    assert "dir" not in sdk.init.call_args.kwargs
    sdk.Settings.assert_called_once_with(disable_code=True, disable_git=True)
    assert tracker.log.call_args_list[0].kwargs == {"step": 0}
    assert tracker.log.call_args_list[0].args[0] == {
        "validation/mlm": 2.0,
        "validation/sources/atlas": 2.0,
    }
    assert tracker.log.call_args_list[1].args[0] == {
        "train/total": 3.0,
        "train/gradient_norms_before_clipping/encoder": 0.5,
    }
    tracker.finish.assert_called_once_with(exit_code=0)


def test_wandb_run_is_finished_as_failed(reporting_config, tmp_path, monkeypatch):
    tracker = Mock()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=Mock(return_value=tracker),
            Settings=Mock(),
        ),
    )
    reporting_config.wandb.enabled = True
    with pytest.raises(RuntimeError, match="training failed"):
        with wandb_run(reporting_config, tmp_path, {}, torch.device("cpu")):
            raise RuntimeError("training failed")
    tracker.finish.assert_called_once_with(exit_code=1)
