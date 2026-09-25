"""Optional Weights & Biases logging helpers.

These are intentionally no-ops unless an active ``wandb`` run exists (i.e. the
pipeline was started with ``wandb.enabled=true`` and ``wandb`` is installed),
so step code can call :func:`log_wandb` unconditionally without importing or
depending on ``wandb`` directly.
"""

from __future__ import annotations

from typing import Any, cast


def log_wandb(metrics: dict[str, Any]) -> None:
    """Log *metrics* to the active W&B run, or do nothing if there isn't one."""
    try:
        import wandb
    except ImportError:
        return
    wandb_api = cast(Any, wandb)
    if wandb_api.run is not None:
        wandb_api.log(metrics)
