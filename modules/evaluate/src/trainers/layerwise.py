"""Layerwise optimization and independent epoch selection."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Literal

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.config.evaluation import TuneConfig
from modules.evaluate.src.config.shared import EvaluationWorkspaceConfig
from modules.evaluate.src.dataset.dataloader import metric_groups_for_ids
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.layerwise import (
    LayerwiseProbeModel,
    run_layerwise_conditions,
)
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.trainers.selection_helpers import (
    build_optimizer,
    build_scheduler,
    checkpoint_state,
    load_checkpoint_state,
    split_decay_parameters,
)
from modules.evaluate.src.trainers.torch import (
    resolve_selection_direction,
    resolve_selection_metric,
)

logger = logging.getLogger(__name__)


def improves(
    score: float, best: float | None, direction: Literal["maximize", "minimize"]
) -> bool:
    """NaN never improves, and any real score beats a missing or NaN best."""
    if math.isnan(score):
        return False
    if best is None or math.isnan(best):
        return True
    return score > best if direction == "maximize" else score < best


def fit_layerwise(
    config: TuneConfig,
    workspace: EvaluationWorkspaceConfig,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any | None,
    accelerator: Accelerator,
    seed: int,
    validation_metric_groups: dict[str, Any] | None = None,
) -> dict[int, dict[str, float]]:
    """Fit standardized heads and return per-layer validation selections."""
    inner = accelerator.unwrap_model(model)
    if not isinstance(inner, LayerwiseProbeModel):
        raise TypeError("Layerwise training requires a LayerwiseProbeModel.")
    inner.fit_standardization(train_loader, accelerator)
    metric = resolve_selection_metric(config, workspace)
    torch.manual_seed(seed)
    for module in inner.heads.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    results = _train_heads(
        config,
        workspace,
        model,
        train_loader,
        val_loader,
        accelerator,
        metric,
        resolve_selection_direction(config, metric),
        validation_metric_groups,
    )
    for layer, (score, epoch) in results.items():
        logger.info(
            "Layer %d selected epoch=%d (val %s=%.4f)", layer, epoch, metric, score
        )
    if results and all((epoch == config.max_epochs for _, epoch in results.values())):
        logger.warning(
            "Every layer selected the final epoch; the probes may be undertrained. Consider raising steps.tune.max_epochs."
        )
    return {
        layer: {"epoch": epoch, "val_score": score}
        for layer, (score, epoch) in results.items()
    }


def fit_layerwise_random_trunk(
    config: TuneConfig,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    inner: LayerwiseProbeModel,
    accelerator: Accelerator,
    seed: int,
    *,
    train_heads: bool,
) -> nn.Module:
    """Fit standardized/trained layer heads over an independently random trunk."""
    token_loaders = (
        training.layerwise.token_dataloaders if training.layerwise is not None else None
    ) or {"train": training.train_dataloader, "validation": training.val_dataloader}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trunk = type(inner.trunk)(deepcopy(inner.trunk.config))
    model = accelerator.prepare(
        LayerwiseProbeModel(
            SimpleNamespace(base_model=trunk, config=trunk.config),
            inner.num_labels,
            inner.task_type,
            inner.config,
            tokenizer=inner.tokenizer,
        ).to(accelerator.device)
    )
    if train_heads:
        fit_layerwise(
            config,
            workspace,
            model,
            token_loaders["train"],
            token_loaders["validation"],
            accelerator,
            seed,
            training.validation_metric_groups,
        )
    else:
        accelerator.unwrap_model(model).fit_standardization(
            token_loaders["train"], accelerator
        )
    return model


def _train_heads(
    config: TuneConfig,
    workspace: EvaluationWorkspaceConfig,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any | None,
    accelerator: Accelerator,
    metric: str,
    direction: Literal["maximize", "minimize"],
    validation_metric_groups: dict[str, Any] | None,
) -> dict[int, tuple[float, int]]:
    """Train with one shared optimizer while tracking stopped heads independently."""
    inner = accelerator.unwrap_model(model)
    assert isinstance(inner, LayerwiseProbeModel)
    task = get_flat_task(workspace.task_type, workspace.num_labels)
    optimizer = build_optimizer(
        config.optimizer,
        split_decay_parameters(inner.heads, config.weight_decay),
        lr=config.learning_rate,
        extra_kwargs=config.optimizer_kwargs,
    )
    updates = math.ceil(len(train_loader) / config.accumulation_steps)
    total_steps = max(1, updates * config.max_epochs * accelerator.num_processes)
    scheduler = build_scheduler(
        config.lr_scheduler_type,
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
        extra_kwargs=config.scheduler_kwargs,
    )
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    patience = config.early_stopping_patience if val_loader is not None else None
    best: dict[int, tuple[float, int]] = {}
    best_states: dict[int, dict[str, Any]] = {}
    stale = dict.fromkeys(inner.layers, 0)

    def stopped(layer: int) -> bool:
        return patience is not None and stale[layer] >= patience

    try:
        for epoch in range(config.max_epochs):
            model.train()
            for batch in train_loader:
                with accelerator.accumulate(model):
                    outputs = model(
                        **{key: value for key, value in batch.items() if key != "id"}
                    )
                    loss = sum(
                        (
                            output.loss * (0 if stopped(layer) else 1)
                            for layer, output in outputs.items()
                        )
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        for layer in inner.layers:
                            if stopped(layer):
                                for parameter in inner.heads[str(layer)].parameters():
                                    parameter.grad = None
                        accelerator.unscale_gradients(optimizer)
                        for head in inner.heads.values():
                            nn.utils.clip_grad_norm_(
                                head.parameters(), config.max_grad_norm
                            )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
            if val_loader is None:
                continue
            results = run_layerwise_conditions(
                model,
                val_loader,
                workspace.task_type,
                accelerator,
                collect_probabilities=metric in task.metrics_requiring_probabilities,
            )["trained"]
            for layer, rows in results.items():
                if stopped(layer):
                    continue
                metric_predictions = task.flatten_for_metrics(rows.predictions)
                metric_labels = task.flatten_for_metrics(rows.labels)
                if validation_metric_groups is not None:
                    group_ids = metric_groups_for_ids(
                        rows.ids, validation_metric_groups, "validation"
                    )
                    score = task.compute_grouped_metrics(
                        metric_predictions,
                        metric_labels,
                        group_ids,
                        [metric],
                        average=config.metric_aggregation,
                    )[metric]
                else:
                    score = task.compute_metrics(
                        metric_predictions,
                        metric_labels,
                        [metric],
                        average=config.metric_aggregation,
                        probabilities=rows.probabilities,
                    )[metric]
                previous = best[layer][0] if layer in best else None
                if improves(score, previous, direction):
                    best[layer] = (score, epoch + 1)
                    best_states[layer] = checkpoint_state(inner.heads[str(layer)])
                    stale[layer] = 0
                else:
                    stale[layer] += 1
            logger.info(
                "Layerwise epoch %d/%d best val %s=%s",
                epoch + 1,
                config.max_epochs,
                metric,
                {layer: score for layer, (score, _) in best.items()},
            )
            if all((stopped(layer) for layer in inner.layers)):
                break
    finally:
        accelerator.free_memory(optimizer, scheduler)
    for layer, state in best_states.items():
        load_checkpoint_state(inner.heads[str(layer)], state)
    return best


def _run_layerwise(
    config: TuneConfig,
    _fit_result: FitResult,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    accelerator: Accelerator,
    seed: int,
) -> Predictor:
    """Fit primary per-layer heads.

    Saved checkpoints contain heads, standardization statistics, and each
    layer's selected epoch, not the frozen trunk.
    """
    assert training.train_dataloader is not None
    if training.embedding is not None and config.batch_size is not None:
        for attribute, shuffle in (
            ("train_dataloader", True),
            ("val_dataloader", False),
        ):
            loader = getattr(training, attribute)
            if loader is not None:
                setattr(
                    training,
                    attribute,
                    accelerator.prepare(
                        DataLoader(
                            loader.dataset,
                            batch_size=config.batch_size,
                            shuffle=shuffle,
                            generator=torch.Generator().manual_seed(seed),
                            collate_fn=loader.collate_fn,
                        )
                    ),
                )
    predictor.model = accelerator.prepare(predictor.model)
    inner = accelerator.unwrap_model(predictor.model)
    assert isinstance(inner, LayerwiseProbeModel)
    selections = fit_layerwise(
        config,
        workspace,
        predictor.model,
        training.train_dataloader,
        training.val_dataloader,
        accelerator,
        seed,
        training.validation_metric_groups,
    )
    _fit_result.layer_selections = selections
    return predictor
