"""Torch probe fitting over token batches or cached embeddings."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast
from torch import nn

from modules.evaluate.src.contracts.artifacts import PreparedArtifacts
from modules.evaluate.src.contracts.config import EvaluationWorkspaceConfig
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.model.inference import run_inference
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.trainers.selection_helpers import (
    build_optimizer,
    build_scheduler,
    checkpoint_state,
    load_checkpoint_state,
    resolve_search_direction,
    split_decay_parameters,
)
from modules.evaluate.src.utils.tracking import log_wandb

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import TuneConfig

logger = logging.getLogger(__name__)


class TorchProbeTrainer:
    """Own the optimizer loop, early stopping, and epoch checkpointing."""

    def __init__(
        self,
        config: TuneConfig,
        train_trunk: bool,
    ) -> None:
        self.config = config
        self.train_trunk = train_trunk

    def resolve_metric_name(self, workspace: EvaluationWorkspaceConfig) -> str:
        return (
            self.config.hyperparameter_search.metric
            or get_flat_task(workspace.task_type).default_metrics()[0]
        )

    def resolve_direction(self, metric_name: str) -> Literal["maximize", "minimize"]:
        return resolve_search_direction(
            metric_name, self.config.hyperparameter_search.direction
        )

    def evaluate_validation(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        metric_name: str,
        *,
        cached_val: tuple[torch.Tensor, torch.Tensor] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Evaluate the selected metric on validation data without stage callbacks."""
        if prepared.val_dataloader is None:
            raise ValueError(
                "hyperparameter_search.enabled=True requires a "
                "val_dataloader, but the Prepare step did not produce one "
                "(no 'validation' split in the dataset)."
            )
        task = get_flat_task(workspace.task_type, workspace.num_labels)
        if cached_val is None:
            cached_val = self._cached_embedding_split(prepared.val_dataloader)
        if cached_val is not None:
            predictions_tensors: list[torch.Tensor] = []
            probabilities_tensors: list[torch.Tensor] = []
            with torch.inference_mode():
                batch_size = batch_size or self._cached_batch_size(
                    prepared.val_dataloader
                )
                for batch in self._cached_batches(
                    cached_val,
                    batch_size,
                    shuffle=False,
                    device=accelerator.device,
                ):
                    outputs = prepared.model(inputs_embeds=batch["inputs_embeds"])
                    predictions_tensors.append(
                        task.extract_predictions(outputs.logits, labels=batch["labels"])
                    )
                    if metric_name in task.metrics_requiring_probabilities:
                        probabilities_tensors.append(
                            task.predicted_probabilities(outputs.logits)
                        )
            predictions = torch.cat(predictions_tensors).cpu().tolist()
            labels = cached_val[1].cpu().tolist()
            probabilities = (
                torch.cat(probabilities_tensors).cpu().tolist()
                if probabilities_tensors
                else None
            )
        else:
            _, predictions, labels, probabilities, _ = run_inference(
                prepared.model,
                prepared.val_dataloader,
                workspace.task_type,
                accelerator,
                collect_probabilities=(
                    metric_name in task.metrics_requiring_probabilities
                ),
            )
        metric_predictions = (
            predictions
            if cached_val is not None and task.is_ragged
            else task.flatten_for_metrics(predictions)
        )
        metric_labels = (
            labels
            if cached_val is not None and task.is_ragged
            else task.flatten_for_metrics(labels)
        )
        if cached_val is not None and workspace.task_type == "token_classification":
            valid = np.asarray(metric_labels) != IGNORE_INDEX
            metric_predictions = np.asarray(metric_predictions)[valid]
            metric_labels = np.asarray(metric_labels)[valid]
        return task.compute_metrics(
            predictions=metric_predictions,
            labels=metric_labels,
            metric_names=[metric_name],
            average=self.config.metric_aggregation,
            probabilities=probabilities,
        )[metric_name]

    def _cached_batch_size(self, dataloader: Any) -> int:
        """Honor the tuning override or inherit the prepared loader's batch size."""
        # Accelerate stores the batch size on the sampler of wrapped loaders.
        return (
            self.config.batch_size
            or getattr(dataloader, "batch_size", None)
            or getattr(getattr(dataloader, "batch_sampler", None), "batch_size", None)
            or 128
        )

    @staticmethod
    def _cached_embedding_split(
        dataloader: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached features in storage, or ``None`` for token batches."""
        dataset = find_embedding_dataset(dataloader)
        if dataset is None:
            return None
        return (
            dataset.embeddings,
            torch.as_tensor(dataset.labels, device=dataset.embeddings.device),
        )

    def _cached_batches(
        self,
        split: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        shuffle: bool,
        device: torch.device,
    ) -> Any:
        embeddings, labels = split
        indices = (
            torch.randperm(embeddings.shape[0], device=embeddings.device)
            if shuffle
            else torch.arange(embeddings.shape[0], device=embeddings.device)
        )
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield {
                "inputs_embeds": embeddings[batch_indices].to(device=device),
                "labels": labels[batch_indices].to(device=device),
            }

    def _set_train_mode(self, model: nn.Module, accelerator: Accelerator) -> None:
        """Put ``model`` in training mode, keeping a frozen trunk in eval mode.

        When ``train_trunk=False`` (the linear-probe setting) the trunk's
        weights are frozen, but a plain ``model.train()`` would still leave the
        trunk's dropout/normalization layers in stochastic training mode. That
        makes the "frozen" representations vary from epoch to epoch, adding
        noise to the probe and hurting reproducibility. Forcing the trunk back
        to eval mode keeps the probed features fixed while the head still
        trains normally.
        """
        model.train()
        if self.train_trunk:
            return
        inner = accelerator.unwrap_model(model)
        base_prefix = getattr(inner, "base_model_prefix", None)
        base = getattr(inner, base_prefix, None) if base_prefix else None
        if base is not None:
            base.eval()

    def train(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        hparams: dict[str, Any],
        trial: Any = None,
        prune_enabled: bool = False,
        hpo_trial: tuple[int, int] | None = None,
    ) -> float | None:
        model = prepared.model
        train_dataloader = prepared.train_dataloader
        assert train_dataloader is not None
        cached_train = self._cached_embedding_split(train_dataloader)
        cached_probe = cached_train is not None

        patience = self.config.early_stopping_patience
        monitor = (patience is not None or prune_enabled) and (
            prepared.val_dataloader is not None
        )
        if patience is not None and prepared.val_dataloader is None:
            logger.warning(
                "early_stopping_patience=%d is set but no validation "
                "dataloader is available (no 'validation' split); early "
                "stopping is disabled for this run.",
                patience,
            )
        metric_name = self.resolve_metric_name(workspace) if monitor else None
        direction = self.resolve_direction(metric_name) if metric_name else "maximize"
        # Keep validation features in storage; transfer each batch when scoring.
        cached_val = (
            self._cached_embedding_split(prepared.val_dataloader) if monitor else None
        )

        param_groups = split_decay_parameters(model, hparams["weight_decay"])
        optimizer = build_optimizer(
            self.config.optimizer,
            param_groups,
            lr=hparams["learning_rate"],
            extra_kwargs=self.config.optimizer_kwargs,
        )
        max_epochs = hparams["max_epochs"]
        batch_size = hparams.get("batch_size") or self._cached_batch_size(
            train_dataloader
        )
        steps_per_epoch = (
            math.ceil(cached_train[0].shape[0] / batch_size)
            if cached_train is not None
            else len(train_dataloader)
        )
        updates_per_epoch = math.ceil(steps_per_epoch / self.config.accumulation_steps)
        # accelerator.prepare() wraps the scheduler so it advances once per
        # process per optimizer step, so the schedule spans that many more steps.
        process_multiplier = 1 if cached_probe else accelerator.num_processes
        total_steps = max(1, updates_per_epoch * max_epochs * process_multiplier)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = build_scheduler(
            self.config.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            extra_kwargs=self.config.scheduler_kwargs,
        )
        if not cached_probe:
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

        best_epoch_score: float | None = None
        best_epoch_state: dict[str, Any] | None = None
        last_epoch_score: float | None = None
        epochs_without_improvement = 0

        training_started = monotonic()
        self._set_train_mode(model, accelerator)
        try:
            for epoch in range(max_epochs):
                epoch_started = monotonic()
                last_progress_logged = epoch_started
                progress_interval = max(1, math.ceil(steps_per_epoch / 10))
                time_check_interval = min(100, progress_interval)
                next_progress_batch = progress_interval
                # Accumulate a batch-size-weighted loss sum and sample count as
                # device tensors and only sync to host once per epoch (below),
                # rather than calling .item() every step, which would force a
                # CUDA sync on every batch. `loss` here is the raw, per-batch
                # mean loss: accelerator.backward() scales it internally for
                # gradient accumulation but does not modify this variable, so
                # weighting by batch size and dividing by total sample count
                # (rather than by step count) reports an accurate mean loss
                # regardless of accumulation or uneven batch sizes.
                running_loss = torch.zeros((), device=accelerator.device)
                running_count = torch.zeros((), device=accelerator.device)
                batches = (
                    self._cached_batches(
                        cached_train,
                        batch_size,
                        shuffle=True,
                        device=accelerator.device,
                    )
                    if cached_train is not None
                    else train_dataloader
                )
                for batch_index, batch in enumerate(batches):
                    completed_batches = batch_index + 1
                    if isinstance(batch, (tuple, list)):
                        inputs_embeds, labels = batch
                        batch = {
                            "inputs_embeds": inputs_embeds,
                            "labels": labels,
                        }
                    else:
                        batch.pop("id", None)
                        batch.pop("residue_lengths", None)

                    accumulation = (
                        nullcontext() if cached_probe else accelerator.accumulate(model)
                    )
                    if cached_probe:
                        accelerator.sync_gradients = (
                            (batch_index + 1) % self.config.accumulation_steps == 0
                            or batch_index + 1 == steps_per_epoch
                        )
                    with accumulation:
                        outputs = model(**batch)
                        loss = outputs.loss
                        # backward() applies accumulation scaling in both
                        # paths. Cached probes use raw optimizers/schedulers,
                        # so update gating must still be explicit below.
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                model.parameters(), self.config.max_grad_norm
                            )
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                    current_batch_size = batch.get(
                        "num_sequences", next(iter(batch.values())).shape[0]
                    )
                    running_loss += loss.detach() * current_batch_size
                    running_count += current_batch_size

                    if (
                        hpo_trial is not None
                        and accelerator.is_main_process
                        and (
                            completed_batches == steps_per_epoch
                            or completed_batches % time_check_interval == 0
                        )
                    ):
                        now = monotonic()
                        elapsed = now - epoch_started
                        if (
                            completed_batches >= next_progress_batch
                            or now - last_progress_logged >= 60
                            or completed_batches == steps_per_epoch
                        ):
                            batches_per_second = completed_batches / max(elapsed, 1e-9)
                            remaining_batches = steps_per_epoch - completed_batches
                            eta = remaining_batches / max(batches_per_second, 1e-9)
                            logger.info(
                                "HPO trial %d/%d | epoch %d/%d | batches "
                                "%d/%d (%.0f%%) | %.1f batches/s | elapsed=%.1fs | "
                                "ETA=%.1fs",
                                hpo_trial[0],
                                hpo_trial[1],
                                epoch + 1,
                                max_epochs,
                                completed_batches,
                                steps_per_epoch,
                                100 * completed_batches / max(steps_per_epoch, 1),
                                batches_per_second,
                                elapsed,
                                eta,
                            )
                            last_progress_logged = now
                            while next_progress_batch <= completed_batches:
                                next_progress_batch += progress_interval

                if cached_probe:
                    mean_loss = (running_loss / running_count.clamp(min=1)).item()
                else:
                    total_loss, total_count = accelerator.gather(
                        (running_loss.unsqueeze(0), running_count.unsqueeze(0))
                    )
                    mean_loss = (
                        total_loss.sum() / total_count.sum().clamp(min=1)
                    ).item()
                logger.info(
                    "Epoch %d/%d | mean loss=%.4f", epoch + 1, max_epochs, mean_loss
                )
                log_wandb({"train/loss": mean_loss, "train/epoch": epoch + 1})

                if not monitor:
                    continue
                assert metric_name is not None
                score = self.evaluate_validation(
                    workspace,
                    prepared,
                    accelerator,
                    metric_name,
                    cached_val=cached_val,
                    batch_size=hparams.get("batch_size"),
                )
                last_epoch_score = score
                self._set_train_mode(model, accelerator)
                logger.info(
                    "Epoch %d/%d | val %s=%.4f",
                    epoch + 1,
                    max_epochs,
                    metric_name,
                    score,
                )
                log_wandb({f"val/{metric_name}": score, "train/epoch": epoch + 1})

                if prune_enabled:
                    if trial is not None:
                        trial.report(score, step=epoch)
                    should_prune_local = (
                        trial.should_prune() if trial is not None else False
                    )
                    should_prune = torch.tensor(
                        int(should_prune_local), device=accelerator.device
                    )
                    if not cached_probe:
                        should_prune = broadcast(should_prune, from_process=0)
                    if bool(should_prune.item()):
                        if hpo_trial is not None and accelerator.is_main_process:
                            logger.info(
                                "HPO trial %d/%d pruned at epoch %d/%d | elapsed=%.1fs",
                                hpo_trial[0],
                                hpo_trial[1],
                                epoch + 1,
                                max_epochs,
                                monotonic() - training_started,
                            )
                        else:
                            logger.info(
                                "Trial pruned at epoch %d/%d", epoch + 1, max_epochs
                            )
                        import optuna

                        raise optuna.TrialPruned()

                improved = best_epoch_score is None or (
                    score > best_epoch_score
                    if direction == "maximize"
                    else score < best_epoch_score
                )
                if improved:
                    best_epoch_score = score
                    epochs_without_improvement = 0
                    if patience is not None:
                        best_epoch_state = checkpoint_state(
                            accelerator.unwrap_model(model)
                        )
                else:
                    epochs_without_improvement += 1

                should_stop = torch.tensor(
                    int(
                        patience is not None and epochs_without_improvement >= patience
                    ),
                    device=accelerator.device,
                )
                if bool(should_stop.item()):
                    logger.info(
                        "Early stopping at epoch %d/%d (no improvement for %d epochs)",
                        epoch + 1,
                        max_epochs,
                        patience,
                    )
                    break

            if patience is not None and best_epoch_state is not None:
                load_checkpoint_state(accelerator.unwrap_model(model), best_epoch_state)
        finally:
            # Must run even if optuna.TrialPruned() propagates out of the
            # loop above, otherwise a pruned trial permanently leaks its
            # optimizer/scheduler in the accelerator's internal tracking.
            accelerator.free_memory(optimizer, scheduler)

        model.eval()
        return best_epoch_score if patience is not None else last_epoch_score
