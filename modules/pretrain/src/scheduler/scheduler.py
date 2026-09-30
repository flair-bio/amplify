from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler as _hf_get_scheduler


class SchedulerConfig(BaseModel):
    """Config for the learning-rate scheduler wrapper."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lr_scheduler_type: str = Field(default="cosine_with_min_lr", alias="type")
    num_warmup_steps: int = Field(default=1000, ge=0)
    num_training_steps: int = Field(default=900000, gt=0)
    scheduler_specific_kwargs: dict[str, Any] | None = Field(
        default_factory=lambda: {"min_lr_rate": 0.1},
        alias="kwargs",
    )


def get_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
) -> LRScheduler:
    """Create a learning rate scheduler using HuggingFace Transformers.

    Thin wrapper around :func:`transformers.get_scheduler` that clamps the
    learning rate to its final value beyond ``num_training_steps``. Some HF
    schedulers (e.g. ``cosine_with_min_lr``) let the LR oscillate past the
    minimum when stepped beyond the configured horizon; the clamping ensures
    the LR stays constant instead.

    Common scheduler types:
        - ``"cosine_with_min_lr"``: Cosine decay with warmup. Supports
          ``min_lr_rate`` in *scheduler_specific_kwargs*.
        - ``"linear"``: Linear decay from peak LR to 0 after warmup.
        - ``"cosine"``: Cosine decay from peak LR to 0 after warmup.
        - ``"warmup_stable_decay"``: Warmup → constant → decay. Supports
          ``num_stable_steps``, ``num_decay_steps``, ``min_lr_ratio``, and
          ``decay_type`` in *scheduler_specific_kwargs*.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        scheduler_config: Validated scheduler config containing scheduler type,
            warmup/training steps, and optional scheduler kwargs.

    Returns:
        A PyTorch LR scheduler.
    """
    lr_scheduler_type = scheduler_config.lr_scheduler_type
    num_warmup_steps = scheduler_config.num_warmup_steps
    num_training_steps = scheduler_config.num_training_steps
    scheduler_specific_kwargs = scheduler_config.scheduler_specific_kwargs

    scheduler = _hf_get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs,
    )

    # Some schedulers (e.g., constant) don't use lr_lambdas — return as-is.
    if not hasattr(scheduler, "lr_lambdas"):
        return scheduler

    def _make_clamped_lambda(
        original_fn: Callable[[int], float], threshold: int
    ) -> Callable[[int], float]:
        """Factory to trap scope cleanly without default-arg hacks."""
        # Precomputing saves math operations (e.g., cosine) during extended steps
        final_factor = original_fn(threshold)
        return lambda step: final_factor if step >= threshold else original_fn(step)

    # Modify in-place to avoid dangling scheduler instances and PyTorch warnings
    scheduler.lr_lambdas = [
        _make_clamped_lambda(lr_lambda, num_training_steps)
        for lr_lambda in scheduler.lr_lambdas
    ]

    return scheduler
