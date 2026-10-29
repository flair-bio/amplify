"""Optimization and per-layer epoch selection for layerwise probes."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import torch
from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.contracts.artifacts import PreparedArtifacts
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
    resolve_search_direction,
    split_decay_parameters,
)

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import (
        EvaluationWorkspaceConfig,
        TuneConfig,
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


class LayerwiseProbeTrainer:
    """Standardize features, train layer heads, and restore each layer's best epoch."""

    def __init__(
        self,
        config: TuneConfig,
    ) -> None:
        self.config = config

    def resolve_metric_name(self, workspace: EvaluationWorkspaceConfig) -> str:
        return (
            self.config.hyperparameter_search.metric
            or get_flat_task(workspace.task_type).default_metrics()[0]
        )

    def resolve_direction(self, metric_name: str) -> Literal["maximize", "minimize"]:
        return resolve_search_direction(
            metric_name, self.config.hyperparameter_search.direction
        )

    def fit(
        self,
        workspace: EvaluationWorkspaceConfig,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any | None,
        accelerator: Accelerator,
        seed: int,
    ) -> dict[int, dict[str, float]]:
        """Fit standardized heads and return per-layer validation selections."""
        inner = accelerator.unwrap_model(model)
        if not isinstance(inner, LayerwiseProbeModel):
            raise TypeError("Layerwise training requires a LayerwiseProbeModel.")
        inner.fit_standardization(train_loader, accelerator)
        metric = self.resolve_metric_name(workspace)
        torch.manual_seed(seed)
        for module in inner.heads.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        results = self._train_heads(
            workspace,
            model,
            train_loader,
            val_loader,
            accelerator,
            metric,
            self.resolve_direction(metric),
        )
        for layer, (score, epoch) in results.items():
            logger.info(
                "Layer %d selected epoch=%d (val %s=%.4f)", layer, epoch, metric, score
            )
        if results and all(
            epoch == self.config.max_epochs for _, epoch in results.values()
        ):
            logger.warning(
                "Every layer selected the final epoch; the probes may be "
                "undertrained. Consider raising steps.tune.max_epochs."
            )
        return {
            layer: {"epoch": epoch, "val_score": score}
            for layer, (score, epoch) in results.items()
        }

    def fit_random_trunk_control(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        inner: LayerwiseProbeModel,
        accelerator: Accelerator,
        seed: int,
        *,
        train_heads: bool,
    ) -> nn.Module:
        """Fit standardized/trained layer heads over an independently random trunk."""
        token_loaders = (
            prepared.layerwise.token_dataloaders
            if prepared.layerwise is not None
            else None
        ) or {
            "train": prepared.train_dataloader,
            "validation": prepared.val_dataloader,
        }
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
            self.fit(
                workspace,
                model,
                token_loaders["train"],
                token_loaders["validation"],
                accelerator,
                seed,
            )
        else:
            accelerator.unwrap_model(model).fit_standardization(
                token_loaders["train"], accelerator
            )
        return model

    def _train_heads(
        self,
        workspace: EvaluationWorkspaceConfig,
        model: nn.Module,
        train_loader: Any,
        val_loader: Any | None,
        accelerator: Accelerator,
        metric: str,
        direction: Literal["maximize", "minimize"],
    ) -> dict[int, tuple[float, int]]:
        """Train with one shared optimizer while tracking stopped heads independently."""
        inner = accelerator.unwrap_model(model)
        assert isinstance(inner, LayerwiseProbeModel)
        task = get_flat_task(workspace.task_type, workspace.num_labels)
        optimizer = build_optimizer(
            self.config.optimizer,
            split_decay_parameters(inner.heads, self.config.weight_decay),
            lr=self.config.learning_rate,
            extra_kwargs=self.config.optimizer_kwargs,
        )
        updates = math.ceil(len(train_loader) / self.config.accumulation_steps)
        total_steps = max(
            1, updates * self.config.max_epochs * accelerator.num_processes
        )
        scheduler = build_scheduler(
            self.config.lr_scheduler_type,
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
            extra_kwargs=self.config.scheduler_kwargs,
        )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
        patience = (
            self.config.early_stopping_patience if val_loader is not None else None
        )
        best: dict[int, tuple[float, int]] = {}
        best_states: dict[int, dict[str, Any]] = {}
        stale = dict.fromkeys(inner.layers, 0)

        def stopped(layer: int) -> bool:
            return patience is not None and stale[layer] >= patience

        try:
            for epoch in range(self.config.max_epochs):
                model.train()
                for batch in train_loader:
                    with accelerator.accumulate(model):
                        outputs = model(
                            **{
                                key: value
                                for key, value in batch.items()
                                if key != "id"
                            }
                        )
                        # Keep stopped heads in the graph for distributed reducers.
                        loss = sum(
                            output.loss * (0 if stopped(layer) else 1)
                            for layer, output in outputs.items()
                        )
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # Prevent shared-optimizer weight decay touching stopped heads.
                            for layer in inner.layers:
                                if stopped(layer):
                                    for parameter in inner.heads[
                                        str(layer)
                                    ].parameters():
                                        parameter.grad = None
                            # Unscale once, then clip each layer independently.
                            accelerator.unscale_gradients(optimizer)
                            for head in inner.heads.values():
                                nn.utils.clip_grad_norm_(
                                    head.parameters(), self.config.max_grad_norm
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
                    collect_probabilities=metric
                    in task.metrics_requiring_probabilities,
                )["trained"]
                for layer, rows in results.items():
                    if stopped(layer):
                        continue
                    score = task.compute_metrics(
                        task.flatten_for_metrics(rows.predictions),
                        task.flatten_for_metrics(rows.labels),
                        [metric],
                        average=self.config.metric_aggregation,
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
                    self.config.max_epochs,
                    metric,
                    {layer: score for layer, (score, _) in best.items()},
                )
                if all(stopped(layer) for layer in inner.layers):
                    break
        finally:
            accelerator.free_memory(optimizer, scheduler)
        for layer, state in best_states.items():
            load_checkpoint_state(inner.heads[str(layer)], state)
        return best
