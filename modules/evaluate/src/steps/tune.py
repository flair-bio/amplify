"""
Train a probe on top of a PLM to evaluate the quality of the representations for a given task.
Optionally hyperparameter search can be performed to find the best probe configuration.
"""

from __future__ import annotations

import copy
import itertools
import logging
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Collection, Literal

import joblib
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast, broadcast_object_list
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.heads import (
    CLASSICAL_HEAD_TYPES,
    EmbeddingProbeHead,
    HeadConfig,
    build_estimator,
    has_classical_search_space,
    sweep_classical_head,
    validate_native_linear_head,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.utils.controls import RANDOMIZED_TRUNK_CONTROLS
from modules.evaluate.src.utils.embed import (
    find_embedding_dataset,
)
from modules.evaluate.src.utils.hpo import (
    OPTIMIZER_REGISTRY,
    build_optimizer,
    build_scheduler,
    checkpoint_state,
    cleanup_between_trials,
    configure_trainable_params,
    create_optuna_study,
    load_checkpoint_state,
    resolve_search_direction,
    sample_candidates,
    split_decay_parameters,
)

if TYPE_CHECKING:
    import optuna
from modules.evaluate.src.utils.inference import run_inference
from modules.evaluate.src.utils.io import (
    write_settings_json,
)
from modules.evaluate.src.utils.layerwise import run_layerwise_conditions
from modules.evaluate.src.utils.tracking import log_wandb

logger = logging.getLogger(__name__)


def _improves(
    score: float, best: float | None, direction: Literal["maximize", "minimize"]
) -> bool:
    """NaN never improves, and any real score beats a missing or NaN best."""
    if math.isnan(score):
        return False
    if best is None or math.isnan(best):
        return True
    return score > best if direction == "maximize" else score < best


class HyperparameterSearchSpaceConfig(BaseModel):
    """Search space options used by hyperparameter optimization."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: list[float] = [5e-4, 1e-4, 5e-5, 1e-5]
    max_epochs: list[int] = [5, 10, 20]
    weight_decay: list[float] = [0.0, 1e-4, 1e-3]
    weight_decay_sampling: Literal["categorical", "log"] = Field(
        "categorical",
        description=(
            "TPE weight-decay sampling: categorical candidates or continuous "
            "log-scale min/max bounds. Random/grid searches retain the listed "
            "candidates. Log sampling requires finite, positive bounds."
        ),
    )
    batch_size: list[PositiveInt] | None = Field(
        None,
        min_length=1,
        description="Optional training batch sizes for cached MLP probes; null keeps tune.batch_size.",
    )
    hidden_size: list[PositiveInt] | None = Field(
        None,
        min_length=1,
        description="Optional hidden widths for cached MLP probes; null keeps head.hidden_size.",
    )

    @model_validator(mode="after")
    def _validate_weight_decay_bounds(self) -> HyperparameterSearchSpaceConfig:
        if self.weight_decay_sampling == "log" and (
            not self.weight_decay
            or any(
                not math.isfinite(value) or value <= 0 for value in self.weight_decay
            )
        ):
            raise ValueError(
                "Log weight_decay sampling requires finite, positive bounds."
            )
        return self


class HyperparameterSearchConfig(BaseModel):
    """Config. for optional hyperparameter search during tuning."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_trials: int = 5
    direction: Literal["maximize", "minimize"] | None = None
    metric: str | None = None
    search_pattern: Literal["bohb", "random", "grid"] = "bohb"
    search_space: HyperparameterSearchSpaceConfig = Field(
        default_factory=HyperparameterSearchSpaceConfig
    )


class TuneConfig(BaseModel):
    """Config for the Tune step: probe training (`tune_probe`) or trunk fine-tuning (`finetune`)."""

    model_config = ConfigDict(extra="forbid")

    head: HeadConfig = Field(default_factory=HeadConfig)

    metric_aggregation: Literal["micro", "macro", "weighted", "samples"] = Field(
        "weighted",
        description=(
            "Averaging used for validation-set model selection (early "
            "stopping, HPO, classical-head sweeps). Auto-synced from "
            "steps.score.metric_aggregation unless set explicitly."
        ),
    )

    # Default (single-run) hyperparameters, used whenever
    # hyperparameter_search.enabled=False.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10
    warmup_ratio: float = Field(
        0.06, description="Fraction of total training steps used for LR warmup."
    )
    max_grad_norm: float = 1.0
    accumulation_steps: int = Field(
        1,
        gt=0,
        description="Number of batches to accumulate before each optimizer step.",
    )
    batch_size: int | None = Field(
        128,
        gt=0,
        description=(
            "Batch size for cached-embedding probe training and validation. "
            "Set to null to use the batch size created by PrepareStep."
        ),
    )
    standardize_features: bool = Field(
        True,
        description=(
            "Standardize ordinary cached MLP/torch_linear probe features with "
            "training statistics. Layerwise probes use their own setting."
        ),
    )
    early_stopping_patience: int | None = Field(
        5,
        description=(
            "Stop a training run early if the validation metric hasn't "
            "improved for this many epochs, restoring the best epoch's "
            "weights."
        ),
    )

    optimizer: str = Field(
        "adamw",
        description=f"Optimizer name, one of: {sorted(OPTIMIZER_REGISTRY)}.",
    )
    optimizer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra constructor kwargs for the chosen optimizer.",
    )
    lr_scheduler_type: str = Field(
        "linear",
        description="Scheduler name passed to transformers.get_scheduler.",
    )
    scheduler_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra scheduler-specific kwargs.",
    )

    hyperparameter_search: HyperparameterSearchConfig = Field(
        default_factory=HyperparameterSearchConfig
    )

    save_best_model: bool = Field(
        True,
        description="Whether to persist the best-performing model and tokenizer.",
    )

    variance_seeds: list[int] = Field(
        default_factory=list,
        description=(
            "Extra seeds to retrain the selected hyperparameters with, purely "
            "to log validation-metric mean/std across training seeds. The "
            "canonical model returned by this step still uses steps.seed; "
            "only cached-embedding probes support this (seeds are cheap there)."
        ),
    )


class TuneStep:
    def __init__(
        self,
        config: TuneConfig,
        mode: Literal["tune_probe", "finetune"] = "tune_probe",
    ) -> None:
        self.config = config
        self.train_trunk = mode == "finetune"

    def run(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
        controls: Collection[str] = (),
    ) -> PreparedArtifacts:
        """Train the probe (and trunk in finetune mode); layerwise runs also fit baselines that *controls* need."""
        logger.info(
            "Running Tune step (%s)...",
            "finetune" if self.train_trunk else "tune_probe",
        )
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.accumulation_steps
            )

        # Guard in case the user misconfigures the pipeline and tries to run TuneStep without a train dataloader (e.g. no 'train' split in the dataset).
        if prepared.train_dataloader is None:
            raise ValueError(
                "TuneStep requires a train_dataloader, but the Prepare step "
                "did not produce one (no 'train' split in the dataset)."
            )
        search = self.config.hyperparameter_search
        if (
            search.enabled
            and (
                search.search_space.hidden_size is not None
                or search.search_space.batch_size is not None
            )
            and (
                self.train_trunk
                or self.config.head.head_type != "mlp"
                or not isinstance(prepared.model, EmbeddingProbeHead)
                or find_embedding_dataset(prepared.train_dataloader) is None
            )
        ):
            dimension = (
                "hidden_size"
                if search.search_space.hidden_size is not None
                else "batch_size"
            )
            raise ValueError(
                f"{dimension} search requires a cached MLP probe in tune_probe mode."
            )
        if isinstance(prepared.model, LayerwiseProbeModel):
            return self._run_layerwise(workspace, prepared, accelerator, seed, controls)

        # If we're using a classical head (e.g. logistic regression, random forest, etc.) we bypass the epoch/optimizer loop entirely and fit it directly on cached pooled embeddings.
        # Embeddings dataloaders were prepared in the previous step.
        if self.config.head.head_type in CLASSICAL_HEAD_TYPES:
            return self._run_classical_head(workspace, prepared, seed, accelerator)

        if not isinstance(prepared.model, nn.Module):
            raise TypeError("Torch probe tuning requires a torch.nn.Module model.")
        # PrepareStep creates a linear probe for cached representations.
        # Rebuild it with the configured MLP width; cached embeddings remain unchanged.
        if self.config.head.head_type == "mlp" and isinstance(
            prepared.model, EmbeddingProbeHead
        ):
            prepared.model = EmbeddingProbeHead(
                prepared.model.embedding_size,
                prepared.model.num_labels,
                prepared.model.task_type,
                probe_hidden_size=self.config.head.hidden_size,
            ).to(accelerator.device)

        model = prepared.model
        embedding_dataset = find_embedding_dataset(prepared.train_dataloader)
        cached_probe = embedding_dataset is not None
        if embedding_dataset is not None and isinstance(model, EmbeddingProbeHead):
            features = embedding_dataset.embeddings
            if self.config.standardize_features:
                if workspace.task_type == "token_classification":
                    valid = torch.as_tensor(embedding_dataset.labels) != IGNORE_INDEX
                    if not valid.all():
                        features = features[valid]
                logger.info(
                    "Standardizing cached probe features using %d training rows.",
                    features.shape[0],
                )
            model.fit_standardization(
                features, enabled=self.config.standardize_features
            )
        if not cached_probe and not self.train_trunk:
            logger.warning(
                "Embedding cache is unavailable; tuning head_type='%s' "
                "without it will recompute frozen-trunk representations "
                "during training.",
                self.config.head.head_type,
            )
        if self.config.head.head_type == "torch_linear" and not cached_probe:
            validate_native_linear_head(model)
        distributed_training = not cached_probe

        # Set requires_grad on the unwrapped model before accelerator.prepare()
        # creates DDP/FSDP wrappers. Those wrappers build gradient-reduction
        # state from the current trainable parameters; changing the flags
        # afterward can leave that state inconsistent and hang backward() in
        # multi-GPU training.
        configure_trainable_params(model, self.train_trunk)
        if distributed_training:
            model = accelerator.prepare(model)
            prepared.model = model

        if self.config.hyperparameter_search.enabled:
            # Hparam search for HF-defined heads or trunk + mlp/linear model
            best_hparams = self._search_hyperparameters(
                workspace, prepared, accelerator, seed
            )
        else:
            # If we're not tuning, we just use configured hparams for a single training run
            best_hparams = {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "max_epochs": self.config.max_epochs,
            }
            self._train(workspace, prepared, accelerator, best_hparams)

        # Width search can replace the head with a different architecture.
        model = prepared.model
        if self.config.variance_seeds and cached_probe:
            self._report_torch_seed_variance(
                workspace, prepared, accelerator, best_hparams
            )

        model.eval()
        if self.config.save_best_model:
            self._save_model(
                workspace, prepared, model, accelerator, best_hparams, seed
            )
        return prepared

    def _run_layerwise(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
        controls: Collection[str],
    ) -> PreparedArtifacts:
        """Fit per-layer heads, plus a random-trunk baseline when a control needs it.

        Saved checkpoints contain heads, standardization statistics, and each
        layer's selected epoch, not the frozen trunk.
        """
        assert prepared.train_dataloader is not None
        if (
            prepared.embedding_backend is not None
            and self.config.batch_size is not None
        ):
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
                                batch_size=self.config.batch_size,
                                shuffle=shuffle,
                                generator=torch.Generator().manual_seed(seed),
                                collate_fn=loader.collate_fn,
                            )
                        ),
                    )
        prepared.model = accelerator.prepare(prepared.model)
        inner = accelerator.unwrap_model(prepared.model)
        assert isinstance(inner, LayerwiseProbeModel)
        selections = self._fit_layerwise(
            workspace,
            prepared.model,
            prepared.train_dataloader,
            prepared.val_dataloader,
            accelerator,
            seed,
        )
        if RANDOMIZED_TRUNK_CONTROLS & set(controls):
            prepared.layerwise_random_model = self._fit_random_trunk_baseline(
                workspace,
                prepared,
                inner,
                accelerator,
                seed,
                "random_trunk" in controls,
            )
        if self.config.save_best_model and accelerator.is_main_process:
            for position, layer in enumerate(inner.layers):
                layer_workspace = get_evaluation_workspace(
                    workspace.config, variant=f"{workspace.variant}-layer_{layer}"
                )
                layer_workspace.setup_directories()
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
                        "tune_config": self.config.model_dump(mode="json"),
                    },
                    layer_workspace.model_path / "probe_head.pt",
                )
        accelerator.wait_for_everyone()
        return prepared

    def _fit_random_trunk_baseline(
        self,
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
            self._fit_layerwise(
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

    def _fit_layerwise(
        self,
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
        inner = accelerator.unwrap_model(model)
        inner.fit_standardization(train_loader, accelerator)
        metric = self._resolve_metric_name(workspace)
        torch.manual_seed(seed)
        for module in inner.heads.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        results = self._train_layerwise_heads(
            workspace,
            model,
            train_loader,
            val_loader,
            accelerator,
            metric,
            self._resolve_direction(metric),
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

    def _train_layerwise_heads(
        self,
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
        inner = accelerator.unwrap_model(model)
        task = get_task(workspace.task_type)
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
                                    for parameter in inner.heads[
                                        str(layer)
                                    ].parameters():
                                        parameter.grad = None
                            # Clip each head separately so one layer's gradient
                            # scale doesn't shrink the others' updates. Unscale
                            # once: GradScaler rejects repeated unscaling.
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
                    if _improves(score, previous, direction):
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

    def _report_torch_seed_variance(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        best_hparams: dict[str, Any],
    ) -> None:
        """Log validation-metric mean/std across ``variance_seeds``, so a
        single training seed's score isn't mistaken for run-to-run variance.
        Diagnostic only: restores the canonical model before returning.
        """
        if prepared.val_dataloader is None:
            return
        model = accelerator.unwrap_model(prepared.model)
        canonical_state = checkpoint_state(model)
        metric_name = self._resolve_metric_name(workspace)
        scores = []
        for extra_seed in self.config.variance_seeds:
            torch.manual_seed(extra_seed)
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self._train(workspace, prepared, accelerator, best_hparams)
            scores.append(
                self._evaluate_validation(
                    workspace,
                    prepared,
                    accelerator,
                    metric_name,
                    batch_size=best_hparams.get("batch_size"),
                )
            )
        load_checkpoint_state(model, canonical_state)
        self._log_seed_variance(metric_name, scores)

    def _log_seed_variance(self, metric_name: str, scores: list[float]) -> None:
        mean, std = float(np.mean(scores)), float(np.std(scores))
        logger.info(
            "Seed variance (%d extra seeds, val %s): mean=%.4f std=%.4f scores=%s",
            len(scores),
            metric_name,
            mean,
            std,
            scores,
        )
        log_wandb(
            {
                f"seed_variance/{metric_name}_mean": mean,
                f"seed_variance/{metric_name}_std": std,
            }
        )

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
        dataloader: Any, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return cached features on *device*, or ``None`` for token batches."""
        dataset = find_embedding_dataset(dataloader)
        if dataset is None:
            return None
        return (
            dataset.embeddings.to(device=device),
            torch.as_tensor(dataset.labels, device=device),
        )

    def _cached_batches(
        self,
        split: tuple[torch.Tensor, torch.Tensor],
        batch_size: int,
        shuffle: bool,
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
                "inputs_embeds": embeddings[batch_indices],
                "labels": labels[batch_indices],
            }

    def _run_classical_head(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        seed: int,
        accelerator: Accelerator,
    ) -> PreparedArtifacts:
        """Fit (optionally searching) a classical, gradient-free head
        directly on cached pooled embeddings, bypassing the epoch/optimizer
        loop used for ``head_type='mlp'`` entirely.

        Requires embedding dataloaders (``steps.prepare.embedding_cache.
        enabled=True``): raises if the train batch doesn't carry
        ``inputs_embeds``, since a classical estimator needs a fixed-size
        feature vector per example, not a token sequence.
        """
        if prepared.train_dataloader is None:
            raise ValueError("A train dataloader is required for a classical head.")
        # Checked via the underlying dataset rather than next(iter(...)):
        # iterating the accelerator-prepared dataloader here would trigger
        # Accelerate's DataLoaderShard RNG resync collective on its first
        # ever iteration for this run, which other ranks (skipping straight
        # to _gather_embeddings's dataset-based read) never join.
        if find_embedding_dataset(prepared.train_dataloader) is None:
            raise ValueError(
                f"head_type='{self.config.head.head_type}' requires "
                "steps.prepare.embedding_cache.enabled=True: classical heads "
                "are fit on fixed-size pooled embeddings, not token batches."
            )

        train_X, train_y = self._gather_embeddings(
            prepared.train_dataloader, accelerator
        )
        head_config = self.config.head
        search_config = self.config.hyperparameter_search
        search_requested = (
            has_classical_search_space(
                head_config.head_type,
                head_config.search_space,
                head_config.hyperparameters,
            )
            or search_config.enabled
        )
        val_X = val_y = None
        if prepared.val_dataloader is not None:
            val_X, val_y = self._gather_embeddings(prepared.val_dataloader, accelerator)
        elif search_requested:
            raise ValueError(
                "Classical probe hyperparameter search requires a validation "
                "split. Add a 'validation' split or use fixed "
                "steps.tune.head.hyperparameters."
            )

        metric_name = self._resolve_metric_name(workspace)
        direction = self._resolve_direction(metric_name)

        if prepared.val_dataloader is not None:
            best_estimator, best_hparams, best_score = sweep_classical_head(
                head_config.head_type,
                workspace.task_type,
                train_X,
                train_y,
                val_X,
                val_y,
                head_config.search_space,
                head_config.hyperparameters,
                metric_name,
                direction,
                search_pattern=(
                    search_config.search_pattern if search_config.enabled else "grid"
                ),
                n_trials=search_config.n_trials if search_config.enabled else None,
                seed=seed,
                average=self.config.metric_aggregation,
            )
            logger.info(
                "Selected head_type=%s hparams=%s val_%s=%.4f",
                head_config.head_type,
                best_hparams,
                metric_name,
                best_score,
            )
        else:
            best_estimator = build_estimator(
                head_config.head_type,
                workspace.task_type,
                head_config.hyperparameters,
                seed=seed,
            )
            best_estimator.fit(train_X, train_y)
            best_hparams = head_config.hyperparameters
            best_score = None
            logger.info(
                "Fitted head_type=%s with fixed hparams=%s on train only (no validation split)",
                head_config.head_type,
                best_hparams,
            )

        if self.config.variance_seeds and prepared.val_dataloader is not None:
            scores = [
                sweep_classical_head(
                    head_config.head_type,
                    workspace.task_type,
                    train_X,
                    train_y,
                    val_X,
                    val_y,
                    search_space={},
                    default_hyperparameters=best_hparams,
                    metric_name=metric_name,
                    direction=direction,
                    search_pattern="grid",
                    seed=extra_seed,
                    average=self.config.metric_aggregation,
                )[2]
                for extra_seed in self.config.variance_seeds
            ]
            self._log_seed_variance(metric_name, scores)

        prepared.model = best_estimator

        if self.config.save_best_model and accelerator.is_main_process:
            # Every rank fit the same estimator on the same gathered data and
            # seed, so only the main process needs to write it out.
            workspace.model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_estimator, workspace.model_path / "head.joblib")
            self._write_tune_settings(
                workspace.model_path / "tune_config.json",
                seed,
                best_hparams,
            )
            logger.info(
                "Saved best %s head to %s", head_config.head_type, workspace.model_path
            )
        accelerator.wait_for_everyone()

        return prepared

    @staticmethod
    def _gather_embeddings(
        dataloader: Any, accelerator: Accelerator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the full (N, H)/(N,) embeddings/labels backing *dataloader*.

        *dataloader* wraps an ``EmbeddingDataset`` that's already fully
        materialized on every rank (computed once, gathered across ranks, by
        ``PrepareStep``/``extract_embeddings``) before ``accelerator.prepare()``
        resharded it across ranks for the (here, unused) epoch/DDP training
        loop. Reading straight off the underlying dataset avoids
        redundantly re-gathering that same data across ranks batch by batch.
        """
        dataset = find_embedding_dataset(dataloader)
        if dataset is not None:
            return (
                dataset.embeddings.detach().cpu().numpy(),
                np.asarray(dataset.labels),
            )

        embeddings, labels = [], []
        for batch in dataloader:
            gathered_embeds, gathered_labels = accelerator.gather_for_metrics(
                (batch["inputs_embeds"], batch["labels"])
            )
            embeddings.append(gathered_embeds.detach().cpu().numpy())
            labels.append(gathered_labels.detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0)

    def _save_model(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        model: nn.Module,
        accelerator: Accelerator,
        best_hparams: dict[str, Any],
        seed: int,
    ) -> None:
        """Persist the current (best) model + tokenizer to ``workspace.model_path``."""
        workspace.model_path.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)

        # Safely gathers the model state dict for FSDP/ZeRO distributed training
        state_dict = accelerator.get_state_dict(model)

        if isinstance(unwrapped_model, EmbeddingProbeHead):
            # No save_pretrained/config.json for a probe head; save its state dict directly.
            if accelerator.is_main_process:
                torch.save(state_dict, workspace.model_path / "probe_head.pt")
                self._write_tune_settings(
                    workspace.model_path / "probe_head_config.json",
                    seed,
                    best_hparams,
                    embedding_size=unwrapped_model.embedding_size,
                    probe_hidden_size=unwrapped_model.probe_hidden_size,
                    standardize_features=unwrapped_model.standardize_features,
                    num_labels=unwrapped_model.num_labels,
                    task_type=unwrapped_model.task_type,
                )
                logger.info("Saved best probe head to %s", workspace.model_path)
            accelerator.wait_for_everyone()
            return

        unwrapped_model.save_pretrained(
            workspace.model_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        if accelerator.is_main_process:
            prepared.tokenizer.save_pretrained(workspace.model_path)
            # Record the tuning settings (and resolved best hyperparameters,
            # which may differ from the config's defaults when
            # hyperparameter_search is enabled) needed to reproduce this
            # checkpoint, since the saved weights alone don't capture them.
            self._write_tune_settings(
                workspace.model_path / "tune_config.json",
                seed,
                best_hparams,
            )
            logger.info("Saved best model and tokenizer to %s", workspace.model_path)
        accelerator.wait_for_everyone()

    def _write_tune_settings(
        self,
        path: Path,
        seed: int,
        best_hyperparameters: dict[str, Any],
        **extra: Any,
    ) -> None:
        write_settings_json(
            path,
            seed,
            self.config.model_dump(mode="json"),
            best_hyperparameters=best_hyperparameters,
            **extra,
        )

    def _resolve_metric_name(self, workspace: EvaluationWorkspace) -> str:
        return (
            self.config.hyperparameter_search.metric
            or get_task(workspace.task_type).default_metrics()[0]
        )

    def _resolve_direction(self, metric_name: str) -> Literal["maximize", "minimize"]:
        search_config = self.config.hyperparameter_search
        return resolve_search_direction(
            metric_name,
            search_config.direction,
        )

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

    def _train(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        hparams: dict[str, Any],
        trial: Any = None,
        prune_enabled: bool = False,
    ) -> float | None:
        model = prepared.model
        train_dataloader = prepared.train_dataloader
        assert train_dataloader is not None
        cached_train = self._cached_embedding_split(
            train_dataloader, accelerator.device
        )
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
        metric_name = self._resolve_metric_name(workspace) if monitor else None
        direction = self._resolve_direction(metric_name) if metric_name else "maximize"
        # Keep cached validation features on the device for this training run.
        cached_val = (
            self._cached_embedding_split(prepared.val_dataloader, accelerator.device)
            if monitor
            else None
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

        self._set_train_mode(model, accelerator)
        try:
            for epoch in range(max_epochs):
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
                    self._cached_batches(cached_train, batch_size, shuffle=True)
                    if cached_train is not None
                    else train_dataloader
                )
                for batch_index, batch in enumerate(batches):
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
                score = self._evaluate_validation(
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

    def _evaluate_validation(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        metric_name: str,
        *,
        cached_val: tuple[torch.Tensor, torch.Tensor] | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Score validation, optionally reusing this training run's device cache."""
        if prepared.val_dataloader is None:
            raise ValueError(
                "hyperparameter_search.enabled=True requires a "
                "val_dataloader, but the Prepare step did not produce one "
                "(no 'validation' split in the dataset)."
            )
        task = get_task(workspace.task_type)
        if cached_val is None:
            cached_val = self._cached_embedding_split(
                prepared.val_dataloader, accelerator.device
            )
        if cached_val is not None:
            predictions_tensors: list[torch.Tensor] = []
            probabilities_tensors: list[torch.Tensor] = []
            with torch.inference_mode():
                batch_size = batch_size or self._cached_batch_size(
                    prepared.val_dataloader
                )
                for batch in self._cached_batches(
                    cached_val, batch_size, shuffle=False
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
        scores = task.compute_metrics(
            predictions=metric_predictions,
            labels=metric_labels,
            metric_names=[metric_name],
            average=self.config.metric_aggregation,
            probabilities=probabilities,
        )
        return scores[metric_name]

    def _probe_search_starters(self) -> list[tuple[int | None, int | None]]:
        """Initial width/batch pairs that cover each configured value."""
        space = self.config.hyperparameter_search.search_space
        if space.hidden_size is None and space.batch_size is None:
            return []
        widths = space.hidden_size or [None]
        batches = space.batch_size or [None]
        return [
            (widths[index % len(widths)], batches[index % len(batches)])
            for index in range(max(len(widths), len(batches)))
        ]

    def _generate_candidates(
        self,
        search_pattern: Literal["bohb", "random", "grid"],
        n_trials: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        space = self.config.hyperparameter_search.search_space

        combos = list(
            itertools.product(
                space.learning_rate,
                space.weight_decay,
                space.max_epochs,
                space.hidden_size or [None],
                space.batch_size or [None],
            )
        )

        if search_pattern == "grid":
            # Grid search must exhaustively cover the space in a
            # deterministic order, not a shuffled/truncated subsample of it.
            if len(combos) > n_trials:
                logger.warning(
                    "Grid search space has %d combinations, which exceeds "
                    "n_trials=%d; running the full grid instead of "
                    "truncating it (grid search should not subsample).",
                    len(combos),
                    n_trials,
                )
            candidates = combos
        else:  # "random"
            if n_trials > len(combos):
                logger.warning(
                    "n_trials=%d exceeds the number of unique combinations "
                    "(%d); running all unique combinations once instead.",
                    n_trials,
                    len(combos),
                )
            if space.hidden_size is None and space.batch_size is None:
                candidates = sample_candidates(combos, n_trials, seed)
            else:
                shuffled = sample_candidates(combos, len(combos), seed)
                starters = [
                    (
                        space.learning_rate[0],
                        space.weight_decay[0],
                        space.max_epochs[0],
                        width,
                        batch,
                    )
                    for width, batch in self._probe_search_starters()
                ]
                candidates = (
                    starters + [item for item in shuffled if item not in starters]
                )[:n_trials]

        return [
            {
                "learning_rate": lr,
                "weight_decay": wd,
                "max_epochs": max_epochs,
                "seed": seed,
                **({"hidden_size": hidden_size} if hidden_size is not None else {}),
                **({"batch_size": batch_size} if batch_size is not None else {}),
            }
            for lr, wd, max_epochs, hidden_size, batch_size in candidates
        ]

    def _search_hyperparameters(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> dict[str, Any]:
        search_config = self.config.hyperparameter_search
        space = search_config.search_space
        metric_name = self._resolve_metric_name(workspace)
        direction = self._resolve_direction(metric_name)

        model = accelerator.unwrap_model(prepared.model)
        initial_state = checkpoint_state(model)
        initial_model = model

        best_score: float | None = None
        best_hparams: dict[str, Any] | None = None
        best_state: dict[str, Any] | None = None
        best_model = model

        def consider(hparams: dict[str, Any], score: float) -> bool:
            nonlocal best_score, best_hparams, best_state, best_model
            is_better = best_score is None or (
                score > best_score if direction == "maximize" else score < best_score
            )
            if is_better:
                best_score, best_hparams = score, hparams
                best_state = checkpoint_state(model)
                best_model = model
            return is_better

        def run_trial(
            hparams: dict[str, Any], trial: Any = None, prune_enabled: bool = False
        ) -> float:
            nonlocal model
            if space.hidden_size is not None:
                assert isinstance(initial_model, EmbeddingProbeHead)
                # Match initialization across trials of the same width, then
                # reset the training RNG below so width cannot alter batch order.
                torch.manual_seed(hparams["seed"])
                model = EmbeddingProbeHead(
                    initial_model.embedding_size,
                    initial_model.num_labels,
                    initial_model.task_type,
                    probe_hidden_size=hparams["hidden_size"],
                    standardize_features=initial_model.standardize_features,
                ).to(initial_model.classifier.weight)
                if initial_model.standardize_features:
                    model.feature_mean.copy_(initial_model.feature_mean)
                    model.feature_std.copy_(initial_model.feature_std)
                prepared.model = model
            if "seed" in hparams:
                torch.manual_seed(hparams["seed"])
                # The train dataloader shuffles via its own torch.Generator
                # (seeded once from the pipeline seed in dataloader.py), not
                # the global RNG above -- reset it here too, or a trial's
                # batch order would depend on how many earlier trials already
                # advanced that generator instead of on hparams["seed"].
                train_generator = getattr(prepared.train_dataloader, "generator", None)
                if train_generator is not None:
                    train_generator.manual_seed(hparams["seed"])
            if space.hidden_size is None:
                load_checkpoint_state(model, initial_state)
            try:
                training_score = self._train(
                    workspace,
                    prepared,
                    accelerator,
                    hparams,
                    trial=trial,
                    prune_enabled=prune_enabled,
                )
                if training_score is not None:
                    return training_score
                return self._evaluate_validation(
                    workspace,
                    prepared,
                    accelerator,
                    metric_name,
                    batch_size=hparams.get("batch_size"),
                )
            finally:
                # Runs on every trial outcome, including a pruned trial
                # (optuna.TrialPruned propagates through this finally block),
                # so a long sweep (100+ trials) doesn't accumulate each
                # trial's optimizer/scheduler/gradient CUDA allocations
                # before the caching allocator would otherwise reclaim them.
                cleanup_between_trials()

        if search_config.search_pattern in ("random", "grid"):
            candidates = self._generate_candidates(
                search_config.search_pattern, search_config.n_trials, seed
            )
            for i, hparams in enumerate(candidates, start=1):
                logger.info("Trial %d/%d | %s", i, len(candidates), hparams)
                score = run_trial(hparams)
                logger.info(
                    "Trial %d/%d | %s=%.4f", i, len(candidates), metric_name, score
                )
                consider(hparams, score)
        else:  # "bohb"

            def objective(trial: "optuna.Trial") -> float:
                # Only the main process queries the Optuna trial: every rank
                # runs its own local, independently-created study (see
                # `create_optuna_study` below), so a non-main rank's
                # `trial.suggest_*` would record that rank's own sampled
                # values against a score actually produced by training with
                # the main process's (broadcast) values below -- a
                # self-inconsistent local trial history that silently
                # corrupts that rank's future suggestions. Only the main
                # process's trial is ever suggested into/reported to/queried
                # for pruning; every other rank's local trial is left
                # untouched (still completes, just without recorded params).
                if accelerator.is_main_process:
                    hparams = {
                        # Sampled continuously (log-uniform) rather than from a
                        # fixed categorical set: BOHB/TPE can only exploit the
                        # shape of the search space for continuous parameters
                        # like learning_rate if it's allowed to draw values
                        # between the configured bounds, not just snap to a
                        # handful of discrete choices.
                        "learning_rate": trial.suggest_float(
                            "learning_rate",
                            min(space.learning_rate),
                            max(space.learning_rate),
                            log=True,
                        ),
                        "weight_decay": self._suggest_weight_decay(trial),
                        "max_epochs": trial.suggest_categorical(
                            "max_epochs", space.max_epochs
                        ),
                        "seed": seed,
                    }
                elif (
                    self._cached_embedding_split(
                        prepared.train_dataloader, accelerator.device
                    )
                    is not None
                ):
                    hparams = {
                        "learning_rate": trial.suggest_float(
                            "learning_rate",
                            min(space.learning_rate),
                            max(space.learning_rate),
                            log=True,
                        ),
                        "weight_decay": self._suggest_weight_decay(trial),
                        "max_epochs": trial.suggest_categorical(
                            "max_epochs", space.max_epochs
                        ),
                        "seed": seed,
                    }
                else:
                    hparams = None
                if hparams is not None:
                    if space.hidden_size is not None:
                        hparams["hidden_size"] = trial.suggest_categorical(
                            "hidden_size", space.hidden_size
                        )
                    if space.batch_size is not None:
                        hparams["batch_size"] = trial.suggest_categorical(
                            "batch_size", space.batch_size
                        )
                if (
                    self._cached_embedding_split(
                        prepared.train_dataloader, accelerator.device
                    )
                    is None
                ):
                    [hparams] = broadcast_object_list([hparams], from_process=0)
                assert hparams is not None  # set by rank 0 above
                score = run_trial(
                    hparams,
                    trial=trial if accelerator.is_main_process else None,
                    prune_enabled=True,
                )
                consider(hparams, score)
                return score

            study = create_optuna_study(direction, seed, search_config.n_trials)
            # Initial trials cover each width and batch within n_trials.
            for hidden_size, batch_size in self._probe_search_starters():
                params = {
                    "learning_rate": space.learning_rate[0],
                    "weight_decay": space.weight_decay[0],
                    "max_epochs": space.max_epochs[0],
                }
                if hidden_size is not None:
                    params["hidden_size"] = hidden_size
                if batch_size is not None:
                    params["batch_size"] = batch_size
                study.enqueue_trial(params)
            study.optimize(objective, n_trials=search_config.n_trials)

        logger.info(
            "Best hyperparameters: %s | best %s=%.4f",
            best_hparams,
            metric_name,
            best_score,
        )
        assert best_hparams is not None and best_state is not None  # >= 1 trial run
        load_checkpoint_state(best_model, best_state)
        if space.hidden_size is not None:
            prepared.model = best_model
        return best_hparams

    def _suggest_weight_decay(self, trial: Any) -> float:
        space = self.config.hyperparameter_search.search_space
        if space.weight_decay_sampling == "log":
            return trial.suggest_float(
                "weight_decay",
                min(space.weight_decay),
                max(space.weight_decay),
                log=True,
            )
        return trial.suggest_categorical("weight_decay", space.weight_decay)
