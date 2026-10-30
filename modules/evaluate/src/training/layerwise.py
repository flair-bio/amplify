"""Layerwise probe fitting: one head per trunk layer, shared optimizer, per-layer epochs.

Extracted from the legacy ``TuneStep`` layerwise path. Functions receive a
``TorchProbeTrainer`` for the tuning configuration and metric resolution; they
never call back into a step object. The saved checkpoints contain heads,
standardization statistics, and each layer's selected epoch, not the frozen trunk.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Collection
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.training.search import improves
from modules.evaluate.src.utils.controls import RANDOMIZED_TRUNK_CONTROLS
from modules.evaluate.src.utils.hpo import (
    build_optimizer,
    build_scheduler,
    checkpoint_state,
    load_checkpoint_state,
    split_decay_parameters,
)
from modules.evaluate.src.utils.layerwise import run_layerwise_conditions

if TYPE_CHECKING:
    from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer

logger = logging.getLogger(__name__)


def run_layerwise(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Accelerator,
    seed: int,
    controls: Collection[str],
) -> PreparedArtifacts:
    """Fit per-layer heads, plus a random-trunk baseline when a control needs it."""
    config = trainer.config
    assert prepared.train_dataloader is not None
    if prepared.embedding_backend is not None and config.batch_size is not None:
        for attribute, shuffle in (
            ("train_dataloader", True),
            ("val_dataloader", False),
        ):
            loader = getattr(prepared, attribute)
            if loader is not None:
                setattr(
                    prepared,
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
    prepared.model = accelerator.prepare(prepared.model)
    inner = accelerator.unwrap_model(prepared.model)
    assert isinstance(inner, LayerwiseProbeModel)
    selections = fit_layerwise(
        trainer,
        workspace,
        prepared.model,
        prepared.train_dataloader,
        prepared.val_dataloader,
        accelerator,
        seed,
    )
    if RANDOMIZED_TRUNK_CONTROLS & set(controls):
        prepared.layerwise_random_model = fit_random_trunk_baseline(
            trainer,
            workspace,
            prepared,
            inner,
            accelerator,
            seed,
            "random_trunk" in controls,
        )
    if config.save_best_model and accelerator.is_main_process:
        save_layerwise_heads(trainer, workspace, inner, selections, seed)
    accelerator.wait_for_everyone()
    return prepared


def save_layerwise_heads(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    inner: LayerwiseProbeModel,
    selections: dict[int, dict[str, float]],
    seed: int,
) -> None:
    """Write one ``probe_head.pt`` per layer under its layer-specific workspace."""
    for position, layer in enumerate(inner.layers):
        layer_workspace = get_evaluation_workspace(
            workspace.config,
            variant=f"{workspace.variant}-layer_{layer}",
            seed=workspace.seed,
        )
        layer_workspace.model_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": inner.heads[str(layer)].state_dict(),
                "feature_mean": inner.feature_mean[position].cpu(),
                "feature_std": inner.feature_std[position].cpu(),
                "target_mean": inner.target_mean.cpu(),
                "target_std": inner.target_std.cpu(),
                "selection": selections.get(layer),
                "layer": layer,
                "model_id": workspace.model_repo_id or workspace.model_name,
                "task_type": workspace.task_type,
                "seed": seed,
                "num_labels": inner.num_labels,
                "layerwise_config": inner.config.model_dump(mode="json"),
                "tune_config": trainer.config.model_dump(mode="json"),
            },
            layer_workspace.model_path / "probe_head.pt",
        )


def fit_random_trunk_baseline(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    inner: LayerwiseProbeModel,
    accelerator: Accelerator,
    seed: int,
    train_heads: bool,
) -> nn.Module:
    """Fit probes on a randomly initialized trunk with the same procedure.

    This separates what pretraining adds from what the architecture alone
    provides. ``random_both`` needs only standardization, not trained heads.
    """
    token_loaders = prepared.layerwise_token_dataloaders or {
        "train": prepared.train_dataloader,
        "validation": prepared.val_dataloader,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trunk = type(inner.trunk)(copy.deepcopy(inner.trunk.config))
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
            trainer,
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


def fit_layerwise(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any | None,
    accelerator: Accelerator,
    seed: int,
) -> dict[int, dict[str, float]]:
    """Standardize features, then train heads and pick each layer's epoch.

    Returns the per-layer selection, which is empty without validation.
    """
    config = trainer.config
    inner = accelerator.unwrap_model(model)
    inner.fit_standardization(train_loader, accelerator)
    metric = trainer.resolve_metric_name(workspace)
    torch.manual_seed(seed)
    for module in inner.heads.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    results = train_layerwise_heads(
        trainer,
        workspace,
        model,
        train_loader,
        val_loader,
        accelerator,
        metric,
        trainer.resolve_direction(metric),
    )
    for layer, (score, epoch) in results.items():
        logger.info(
            "Layer %d selected epoch=%d (val %s=%.4f)", layer, epoch, metric, score
        )
    if results and all(epoch == config.max_epochs for _, epoch in results.values()):
        logger.warning(
            "Every layer selected the final epoch; the probes may be "
            "undertrained. Consider raising steps.tune.max_epochs."
        )
    return {
        layer: {"epoch": epoch, "val_score": score}
        for layer, (score, epoch) in results.items()
    }


def train_layerwise_heads(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    model: nn.Module,
    train_loader: Any,
    val_loader: Any | None,
    accelerator: Accelerator,
    metric: str,
    direction: Literal["maximize", "minimize"],
) -> dict[int, tuple[float, int]]:
    """Train all heads with one shared optimizer, selecting epochs per layer.

    Patience and best states are tracked per layer. Returns each layer's
    best ``(score, epoch)`` with that state restored; without validation,
    heads keep their final state.
    """
    config = trainer.config
    inner = accelerator.unwrap_model(model)
    task = get_task(workspace.task_type, workspace.num_labels)
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
                    # Stopped heads keep a zero-weighted loss so DDP still
                    # sees every parameter used in the shared trunk pass.
                    loss = sum(
                        output.loss * (0 if stopped(layer) else 1)
                        for layer, output in outputs.items()
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # The optimizer is shared, so clear stopped-head
                        # gradients to prevent weight decay updating them.
                        for layer in inner.layers:
                            if stopped(layer):
                                for parameter in inner.heads[str(layer)].parameters():
                                    parameter.grad = None
                        # Clip each head separately so one layer's gradient
                        # scale doesn't shrink the others' updates. Unscale
                        # once: GradScaler rejects repeated unscaling.
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
                score = task.compute_metrics(
                    task.flatten_for_metrics(rows.predictions),
                    task.flatten_for_metrics(rows.labels),
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
            if all(stopped(layer) for layer in inner.layers):
                break
    finally:
        accelerator.free_memory(optimizer, scheduler)
    for layer, state in best_states.items():
        load_checkpoint_state(inner.heads[str(layer)], state)
    return best
