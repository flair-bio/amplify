"""Hyperparameter-tuning utilities shared by the Tune step.

Optimizer/scheduler construction, trainable-parameter freezing, and
checkpoint helpers used to train and search over probe hyperparameters.
"""

from __future__ import annotations

import gc
import logging
import random
from typing import Any, Literal, Sequence, TypeVar

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

logger = logging.getLogger(__name__)

# Optimizer name -> torch.optim class. Covers the common cases used for
# probing/fine-tuning; any additional constructor kwargs (e.g. momentum for
# 'sgd', betas for 'adamw') can be passed via TuneConfig.optimizer_kwargs.
OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
    "adafactor": torch.optim.Adafactor,
}

# Metrics for which a lower value is better
LOWER_IS_BETTER_METRICS = frozenset({"mse", "mae", "pseudo_perplexity"})
TPE_MIN_STARTUP_TRIALS = 3
TPE_MAX_STARTUP_TRIALS = 5
TPE_STARTUP_TRIAL_DIVISOR = 3
PRUNER_WARMUP_STEPS = 3

SearchDirection = Literal["maximize", "minimize"]
Candidate = TypeVar("Candidate")


def resolve_search_direction(
    metric_name: str,
    configured_direction: SearchDirection | None,
) -> SearchDirection:
    """Choose a metric-appropriate search direction unless overridden."""
    if configured_direction is None:
        return "minimize" if metric_name in LOWER_IS_BETTER_METRICS else "maximize"
    return configured_direction


def sample_candidates(
    candidates: Sequence[Candidate], n_trials: int, seed: int
) -> list[Candidate]:
    """Deterministically sample unique finite-grid candidates without replacement."""
    sampled = list(candidates)
    random.Random(seed).shuffle(sampled)
    return sampled[:n_trials]


def create_optuna_study(direction: SearchDirection, seed: int, n_trials: int) -> Any:
    """Create the seeded Optuna study shared by neural and classical HPO.

    TPE's default ``n_startup_trials=10`` would make a typical 15-trial budget
    mostly random search, so startup is scaled to the budget. ``MedianPruner``
    is used instead of Hyperband because ``num_epochs`` is itself a search
    dimension, so trials do not share a common resource ladder.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for search_pattern='bohb'. Install the "
            "'evaluate' extra or select 'grid'/'random'."
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=max(
                TPE_MIN_STARTUP_TRIALS,
                min(TPE_MAX_STARTUP_TRIALS, n_trials // TPE_STARTUP_TRIAL_DIVISOR),
            ),
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_STEPS),
    )


def build_optimizer(
    name: str,
    param_groups: list[dict[str, Any]],
    lr: float,
    extra_kwargs: dict[str, Any],
) -> Optimizer:
    """Instantiate an optimizer by name from :data:`OPTIMIZER_REGISTRY`.

    ``param_groups`` is expected to already carry a per-group
    ``weight_decay`` (see :func:`split_decay_parameters`), so only ``lr``
    is supplied as a shared default here.
    """
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unsupported optimizer '{name}'. Supported: {sorted(OPTIMIZER_REGISTRY)}."
        )
    kwargs: dict[str, Any] = {"lr": lr, **extra_kwargs}
    return OPTIMIZER_REGISTRY[key](param_groups, **kwargs)


def split_decay_parameters(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split trainable parameters into weight-decay and no-decay groups.

    Biases and normalization-layer weights/biases (identified by having
    <= 1 dimensions, e.g. LayerNorm/BatchNorm weight and any bias) are
    excluded from weight decay, following common practice.
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return param_groups


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    extra_kwargs: dict[str, Any],
) -> LRScheduler:
    """Instantiate an LR scheduler by name."""
    return get_scheduler(
        name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=extra_kwargs or None,
    )


def checkpoint_state(model: nn.Module) -> dict[str, Any]:
    """Copy only trainable model weights to CPU to avoid host OOM with large frozen trunks."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_checkpoint_state(model: nn.Module, state: dict[str, Any]) -> None:
    """Load a `checkpoint_state` dict back into ``model``.

    Uses ``strict=False`` because ``state`` only contains the trainable
    subset of parameters (frozen params are legitimately "missing"), but
    still asserts there are no *unexpected* keys, which would indicate a
    real bug (e.g. the state was captured from a differently-named model).
    """
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys when restoring checkpointed state: "
            f"{result.unexpected_keys}"
        )


def cleanup_between_trials() -> None:
    """Force a Python GC pass and release cached CUDA memory back to the
    allocator between hyperparameter-search trials.

    Each trial builds its own optimizer/scheduler and (for the checkpointed
    early-stopping/hyperparameter-search paths) copies of the trainable
    parameters via ``checkpoint_state``; over a long sweep (100+ trials)
    these short-lived tensors can accumulate as unreferenced-but-not-yet-
    collected CUDA allocations across trials, since PyTorch's caching
    allocator normally holds onto freed memory rather than returning it to
    the driver. Without this, a long sweep can fragment/exhaust GPU memory
    partway through even though each individual trial fits comfortably on
    its own. ``torch.cuda.empty_cache()`` is a no-op (and safe to call) when
    no CUDA device is available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def configure_trainable_params(model: nn.Module, train_trunk: bool) -> None:
    """Freeze/unfreeze the trunk vs. head parameters of ``model`` in place.

    Uses `base_model_prefix` to determine trunk parameters by name instead of
    relying on `id()`, preventing accidental freezing of tied head weights.
    """
    base_prefix = getattr(model, "base_model_prefix", None)

    if base_prefix is None:
        if train_trunk:
            logger.warning(
                "Could not resolve a distinct trunk submodule via "
                "model.base_model_prefix; leaving all parameters trainable "
                "(mode='finetune' will have no effect)."
            )
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        is_trunk = name == base_prefix or name.startswith(f"{base_prefix}.")
        param.requires_grad = train_trunk or not is_trunk

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s/%s (train_trunk=%s)",
        f"{n_trainable:,}",
        f"{n_total:,}",
        train_trunk,
    )
