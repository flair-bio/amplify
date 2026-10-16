"""Unit tests for the LR scheduler wrapper (warmup and the post-horizon clamp)."""

from __future__ import annotations

import pytest
import torch

from modules.pretrain.src.scheduler.scheduler import SchedulerConfig, get_scheduler

PEAK_LR = 1e-3


def lr_trajectory(config: SchedulerConfig, steps: int) -> list[float]:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK_LR)
    scheduler = get_scheduler(optimizer, config)
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


@pytest.fixture
def config():
    return SchedulerConfig(
        type="cosine_with_min_lr",
        num_warmup_steps=10,
        num_training_steps=100,
        kwargs={"min_lr_rate": 0.1},
    )


def test_warmup_ramps_from_zero_to_peak(config):
    lrs = lr_trajectory(config, steps=11)

    assert lrs[0] == 0.0
    assert lrs[10] == pytest.approx(PEAK_LR)
    assert lrs[:11] == sorted(lrs[:11])


def test_lr_is_held_at_its_final_value_past_the_horizon(config):
    # Covers runs where trainer.max_steps exceeds scheduler.num_training_steps.
    lrs = lr_trajectory(config, steps=300)

    assert lrs[100] == pytest.approx(0.1 * PEAK_LR)
    assert all(lr == lrs[100] for lr in lrs[100:])


def test_yaml_aliases_populate_the_config():
    config = SchedulerConfig.model_validate(
        {"type": "linear", "num_training_steps": 5, "kwargs": None}
    )

    assert config.lr_scheduler_type == "linear"
    assert config.scheduler_specific_kwargs is None
