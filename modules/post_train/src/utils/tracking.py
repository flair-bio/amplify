"""Optional Weights & Biases run lifecycle and metric logging."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import torch

from modules.post_train.src.config import PostTrainingConfig
from modules.post_train.src.trainers.latent_alignment import capture_rng, restore_rng


def wandb_metrics(row: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}

    def flatten(prefix: str, values: Any) -> None:
        if isinstance(values, dict):
            for key, value in values.items():
                flatten(f"{prefix}/{key}", value)
        elif isinstance(values, (int, float)):
            result[prefix] = values

    for key, value in row.items():
        if key != "step":
            flatten("validation" if key == "validation" else f"train/{key}", value)
    return result


def log_wandb(
    metrics: dict[str, Any], *, step: int | None = None, device: torch.device
) -> None:
    """Log metrics to an active W&B run, or do nothing if there isn't one."""
    try:
        import wandb
    except ImportError:
        return
    wandb_api = cast(Any, wandb)
    if wandb_api.run is not None:
        state = capture_rng(device)
        try:
            wandb_api.log(metrics, step=step)
        finally:
            restore_rng(state, device)


@contextmanager
def wandb_run(
    config: PostTrainingConfig,
    output: Path,
    resolved: dict[str, Any],
    device: torch.device,
) -> Iterator[None]:
    tracker = None
    options = config.wandb
    if options.enabled:
        state = capture_rng(device)
        try:
            import wandb

            tracker = wandb.init(
                project=options.project,
                entity=options.entity,
                name=options.run_name or output.name,
                tags=options.tags,
                config=resolved,
                settings=wandb.Settings(disable_code=True, disable_git=True),
            )
        finally:
            restore_rng(state, device)
    try:
        yield
    except BaseException:
        if tracker is not None:
            tracker.finish(exit_code=1)
        raise
    else:
        if tracker is not None:
            tracker.finish(exit_code=0)
