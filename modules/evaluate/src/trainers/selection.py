"""
Train a probe on top of a PLM to evaluate the quality of the representations for a given task.
Optionally hyperparameter search can be performed to find the best probe configuration.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import replace
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.contracts.artifacts import (
    FitResult,
    PreparedArtifacts,
)
from modules.evaluate.src.contracts.config import (
    CLASSICAL_HEAD_TYPES,
    EvaluationWorkspaceConfig,
    TuneConfig,
)
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingDataset,
    find_embedding_dataset,
)
from modules.evaluate.src.model.heads import (
    EmbeddingProbeHead,
    build_estimator,
    has_classical_search_space,
    validate_native_linear_head,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX
from modules.evaluate.src.trainers.classical import ClassicalProbeTrainer
from modules.evaluate.src.trainers.layerwise import LayerwiseProbeTrainer
from modules.evaluate.src.trainers.selection_helpers import (
    checkpoint_state,
    cleanup_between_trials,
    configure_trainable_params,
    create_optuna_study,
    load_checkpoint_state,
    sample_candidates,
)
from modules.evaluate.src.trainers.torch import TorchProbeTrainer

if TYPE_CHECKING:
    import optuna
logger = logging.getLogger(__name__)


class PredictorSelection:
    def __init__(
        self,
        config: TuneConfig,
        mode: Literal["tune_probe", "finetune"] = "tune_probe",
    ) -> None:
        self.config = config
        self.train_trunk = mode == "finetune"
        self._torch_trainer = TorchProbeTrainer(config, self.train_trunk)
        self._layerwise_trainer = LayerwiseProbeTrainer(config)
        self.selected_hyperparameters: dict[str, Any] = {}
        self.layer_selections: dict[int, dict[str, float]] = {}
        self.seed_variance: dict[str, list[float]] = {}
        self.selection_metric: str | None = None
        self.selection_score: float | None = None

    def run(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
    ) -> PreparedArtifacts:
        """Fit the primary probe, and the trunk in finetune mode."""
        logger.info(
            "Running predictor fitting (%s)...",
            "finetune" if self.train_trunk else "tune_probe",
        )
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.accumulation_steps
            )

        # Keep the fitting service safe when called independently of the stage.
        if prepared.train_dataloader is None:
            raise ValueError(
                "Predictor fitting requires a train_dataloader, but the Prepare step "
                "did not produce one (no 'train' split in the dataset)."
            )
        search = self.config.hyperparameter_search
        if (
            search.enabled
            and (
                search.search_space.hidden_size is not None
                or search.search_space.hidden_depth is not None
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
                else (
                    "hidden_depth"
                    if search.search_space.hidden_depth is not None
                    else "batch_size"
                )
            )
            raise ValueError(
                f"{dimension} search requires a cached MLP probe in tune_probe mode."
            )
        if self.config.refit_on_train_validation:
            if (
                self.train_trunk
                or isinstance(prepared.model, LayerwiseProbeModel)
                or find_embedding_dataset(prepared.train_dataloader) is None
                or (
                    self.config.head.head_type not in CLASSICAL_HEAD_TYPES
                    and not isinstance(prepared.model, EmbeddingProbeHead)
                )
            ):
                raise ValueError(
                    "refit_on_train_validation requires an ordinary cached probe "
                    "in tune_probe mode; live/native and layerwise refit are not supported."
                )
            for name, loader in (
                ("train", prepared.train_dataloader),
                ("validation", prepared.val_dataloader),
            ):
                dataset = find_embedding_dataset(loader)
                if dataset is None or not len(dataset):
                    raise ValueError(
                        "refit_on_train_validation requires nonempty train and "
                        f"validation embedding datasets; missing or empty {name} split."
                    )
            missing_refit_splits = set(prepared.refit_token_dataloaders) - set(
                prepared.refit_dataloaders
            )
            if missing_refit_splits:
                raise ValueError(
                    "Final-refit splits were not cached as embeddings: "
                    f"{sorted(missing_refit_splits)}."
                )
            for name, loader in prepared.refit_dataloaders.items():
                dataset = find_embedding_dataset(loader)
                if dataset is None or not len(dataset):
                    raise ValueError(
                        "refit_on_train_validation requires nonempty final-refit "
                        f"embedding datasets; missing or empty {name} split."
                    )
        if isinstance(prepared.model, LayerwiseProbeModel):
            return self._run_layerwise(workspace, prepared, accelerator, seed)

        # If we're using a classical head (e.g. logistic regression, random forest, etc.) we bypass the epoch/optimizer loop entirely and fit it directly on cached pooled embeddings.
        # Embeddings dataloaders were prepared in the previous step.
        if self.config.head.head_type in CLASSICAL_HEAD_TYPES:
            self._run_classical_head(workspace, prepared, seed, accelerator)
            return prepared

        if not isinstance(prepared.model, nn.Module):
            raise TypeError("Torch probe tuning requires a torch.nn.Module model.")
        # PrepareStep creates a linear probe for cached representations.
        # Rebuild the configured MLP architecture; cached embeddings remain unchanged.
        if self.config.head.head_type == "mlp" and isinstance(
            prepared.model, EmbeddingProbeHead
        ):
            prepared.model = EmbeddingProbeHead(
                prepared.model.embedding_size,
                prepared.model.num_labels,
                prepared.model.task_type,
                probe_hidden_size=self.config.head.hidden_size,
                probe_hidden_depth=self.config.head.hidden_depth,
            ).to(accelerator.device)

        model = prepared.model
        embedding_dataset = find_embedding_dataset(prepared.train_dataloader)
        cached_probe = embedding_dataset is not None
        if embedding_dataset is not None and isinstance(model, EmbeddingProbeHead):
            features = embedding_dataset.embeddings
            row_mask = None
            if self.config.standardize_features:
                if workspace.task_type == "token_classification":
                    row_mask = torch.as_tensor(embedding_dataset.labels) != IGNORE_INDEX
                    num_rows = int(row_mask.sum().item())
                else:
                    num_rows = features.shape[0]
                logger.info(
                    "Standardizing cached probe features using %d training rows.",
                    num_rows,
                )
            model.fit_standardization(
                features,
                enabled=self.config.standardize_features,
                row_mask=row_mask,
                show_progress=accelerator.is_main_process,
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
            if not self.config.refit_on_train_validation:
                self._torch_trainer.train(
                    workspace, prepared, accelerator, best_hparams
                )
        self.selected_hyperparameters = dict(best_hparams)

        # Architecture search can replace the head.
        if self.config.variance_seeds and cached_probe:
            self._report_torch_seed_variance(
                workspace, prepared, accelerator, best_hparams
            )
        self._finalize_torch_probe(workspace, prepared, accelerator, best_hparams, seed)
        return prepared

    def _refit_cached_probe(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        hparams: dict[str, Any],
        seed: int,
    ) -> None:
        """Fit a fresh selected head on development data, retaining split loaders."""
        train = find_embedding_dataset(prepared.train_dataloader)
        validation = find_embedding_dataset(prepared.val_dataloader)
        selected = accelerator.unwrap_model(prepared.model)
        assert train is not None and validation is not None
        assert isinstance(selected, EmbeddingProbeHead)
        development_splits = [train, validation]
        for name, loader in prepared.refit_dataloaders.items():
            extra = find_embedding_dataset(loader)
            if extra is None:
                raise ValueError(
                    f"Final-refit split '{name}' has no cached embeddings."
                )
            development_splits.append(extra)
        development = EmbeddingDataset(
            torch.cat([split.embeddings for split in development_splits]),
            [identifier for split in development_splits for identifier in split.ids],
            torch.cat([torch.as_tensor(split.labels) for split in development_splits]),
        )
        torch.manual_seed(seed)
        fresh = EmbeddingProbeHead(
            selected.embedding_size,
            selected.num_labels,
            selected.task_type,
            probe_hidden_size=selected.probe_hidden_size,
            probe_hidden_depth=selected.probe_hidden_depth,
        ).to(selected.classifier.weight)
        row_mask = None
        if workspace.task_type == "token_classification":
            row_mask = (
                torch.as_tensor(
                    development.labels, device=development.embeddings.device
                )
                != IGNORE_INDEX
            )
        fresh.fit_standardization(
            development.embeddings,
            enabled=self.config.standardize_features,
            row_mask=row_mask,
            show_progress=accelerator.is_main_process,
        )
        loader = DataLoader(
            development,
            batch_size=hparams.get("batch_size")
            or self._torch_trainer._cached_batch_size(prepared.train_dataloader),
        )
        refit_prepared = replace(
            prepared,
            model=fresh,
            train_dataloader=loader,
            val_dataloader=None,
            test_dataloader=None,
        )
        logger.info(
            "Refitting fresh cached probe on %d rows from %s for %d epochs.",
            len(development),
            self._final_fit_split(prepared),
            hparams["max_epochs"],
        )
        trainer = TorchProbeTrainer(
            self.config.model_copy(update={"early_stopping_patience": None}),
            train_trunk=False,
        )
        torch.manual_seed(seed)
        trainer.train(workspace, refit_prepared, accelerator, hparams)
        prepared.model = fresh

    def _finalize_torch_probe(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        best_hparams: dict[str, Any],
        seed: int,
    ) -> None:
        """Apply the shared selected-model refit, evaluation, and persistence handoff."""
        if self.config.refit_on_train_validation:
            self._refit_cached_probe(
                workspace, prepared, accelerator, best_hparams, seed
            )
        model = prepared.model
        if not isinstance(model, nn.Module):
            raise TypeError("Torch probe finalization requires a torch model.")
        model.eval()

    def _run_layerwise(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> PreparedArtifacts:
        """Fit primary per-layer heads.

        Saved checkpoints contain heads, standardization statistics, and each
        layer's selected epoch, not the frozen trunk.
        """
        assert prepared.train_dataloader is not None
        if prepared.embedding is not None and self.config.batch_size is not None:
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
        selections = self._layerwise_trainer.fit(
            workspace,
            prepared.model,
            prepared.train_dataloader,
            prepared.val_dataloader,
            accelerator,
            seed,
        )
        self.layer_selections = selections
        return prepared

    def _report_torch_seed_variance(
        self,
        workspace: EvaluationWorkspaceConfig,
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
        metric_name = self._torch_trainer.resolve_metric_name(workspace)
        scores = []
        for extra_seed in self.config.variance_seeds:
            torch.manual_seed(extra_seed)
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            self._torch_trainer.train(workspace, prepared, accelerator, best_hparams)
            scores.append(
                self._torch_trainer.evaluate_validation(
                    workspace,
                    prepared,
                    accelerator,
                    metric_name,
                    batch_size=best_hparams.get("batch_size"),
                )
            )
        load_checkpoint_state(model, canonical_state)
        self.seed_variance[metric_name] = scores
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

    def _run_classical_head(
        self,
        workspace: EvaluationWorkspaceConfig,
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

        metric_name = self._torch_trainer.resolve_metric_name(workspace)
        direction = self._torch_trainer.resolve_direction(metric_name)

        if prepared.val_dataloader is not None:
            best_estimator, best_hparams, best_score = (
                ClassicalProbeTrainer.fit_candidates(
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
                        search_config.search_pattern
                        if search_config.enabled
                        else "grid"
                    ),
                    n_trials=(
                        search_config.n_trials if search_config.enabled else None
                    ),
                    seed=seed,
                    average=self.config.metric_aggregation,
                    num_labels=workspace.num_labels,
                    is_main_process=accelerator.is_main_process,
                )
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
                ClassicalProbeTrainer.fit_candidates(
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
                    num_labels=workspace.num_labels,
                    is_main_process=accelerator.is_main_process,
                )[2]
                for extra_seed in self.config.variance_seeds
            ]
            self.seed_variance[metric_name] = scores
            self._log_seed_variance(metric_name, scores)

        if self.config.refit_on_train_validation:
            assert val_X is not None and val_y is not None
            final_features = [train_X, val_X]
            final_labels = [train_y, val_y]
            for loader in prepared.refit_dataloaders.values():
                extra_X, extra_y = self._gather_embeddings(loader, accelerator)
                final_features.append(extra_X)
                final_labels.append(extra_y)
            best_estimator = build_estimator(
                head_config.head_type, workspace.task_type, best_hparams, seed=seed
            )
            best_estimator.fit(
                np.concatenate(final_features), np.concatenate(final_labels)
            )
            logger.info(
                "Refitted fresh %s estimator on %d rows from %s.",
                head_config.head_type,
                sum(len(labels) for labels in final_labels),
                self._final_fit_split(prepared),
            )
        prepared.model = best_estimator
        self.selected_hyperparameters = dict(best_hparams)
        self.selection_metric = metric_name
        self.selection_score = best_score

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

    def _final_fit_split(self, prepared: PreparedArtifacts) -> str:
        if not self.config.refit_on_train_validation:
            return "train"
        return "+".join(["train", "validation", *prepared.refit_dataloaders.keys()])

    def _probe_search_starters(
        self,
    ) -> list[tuple[int | None, int | None, int | None]]:
        """Initial width/batch/depth combinations covering each configured value."""
        space = self.config.hyperparameter_search.search_space
        if (
            space.hidden_size is None
            and space.batch_size is None
            and space.hidden_depth is None
        ):
            return []
        widths = space.hidden_size or [None]
        batches = space.batch_size or [None]
        depths = space.hidden_depth or [None]
        return [
            (
                widths[index % len(widths)],
                batches[index % len(batches)],
                depths[index % len(depths)],
            )
            for index in range(max(len(widths), len(batches), len(depths)))
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
                space.hidden_depth or [None],
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
            if not self._probe_search_starters():
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
                        depth,
                    )
                    for width, batch, depth in self._probe_search_starters()
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
                **({"hidden_depth": hidden_depth} if hidden_depth is not None else {}),
            }
            for lr, wd, max_epochs, hidden_size, batch_size, hidden_depth in candidates
        ]

    def _search_hyperparameters(
        self,
        workspace: EvaluationWorkspaceConfig,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> dict[str, Any]:
        search_config = self.config.hyperparameter_search
        space = search_config.search_space
        metric_name = self._torch_trainer.resolve_metric_name(workspace)
        direction = self._torch_trainer.resolve_direction(metric_name)
        if accelerator.is_main_process:
            logger.info(
                "Starting hyperparameter tuning | pattern=%s | trial budget=%d | "
                "metric=%s (%s)",
                search_config.search_pattern,
                search_config.n_trials,
                metric_name,
                direction,
            )

        model = accelerator.unwrap_model(prepared.model)
        initial_state = checkpoint_state(model)
        initial_model = model
        architecture_search = (
            space.hidden_size is not None or space.hidden_depth is not None
        )

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
            hparams: dict[str, Any],
            *,
            trial_number: int,
            trial_total: int,
            trial: Any = None,
            prune_enabled: bool = False,
        ) -> float:
            nonlocal model
            trial_started = monotonic()
            if accelerator.is_main_process:
                logger.info(
                    "HPO trial %d/%d started | params=%s",
                    trial_number,
                    trial_total,
                    hparams,
                )
            if architecture_search:
                assert isinstance(initial_model, EmbeddingProbeHead)
                # Match initialization for the same architecture, then reset
                # the training RNG so architecture cannot alter batch order.
                torch.manual_seed(hparams["seed"])
                model = EmbeddingProbeHead(
                    initial_model.embedding_size,
                    initial_model.num_labels,
                    initial_model.task_type,
                    probe_hidden_size=hparams.get(
                        "hidden_size", initial_model.probe_hidden_size
                    ),
                    probe_hidden_depth=hparams.get(
                        "hidden_depth", initial_model.probe_hidden_depth
                    ),
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
            if not architecture_search:
                load_checkpoint_state(model, initial_state)
            try:
                training_score = self._torch_trainer.train(
                    workspace,
                    prepared,
                    accelerator,
                    hparams,
                    trial=trial,
                    prune_enabled=prune_enabled,
                    hpo_trial=(trial_number, trial_total),
                )
                if training_score is not None:
                    score = training_score
                else:
                    score = self._torch_trainer.evaluate_validation(
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
            is_best = consider(hparams, score)
            if accelerator.is_main_process:
                logger.info(
                    "HPO trial %d/%d complete | val_%s=%.4f | "
                    "best_%s=%.4f | new_best=%s | elapsed=%.1fs",
                    trial_number,
                    trial_total,
                    metric_name,
                    score,
                    metric_name,
                    best_score,
                    is_best,
                    monotonic() - trial_started,
                )
            return score

        if search_config.search_pattern in ("random", "grid"):
            candidates = self._generate_candidates(
                search_config.search_pattern, search_config.n_trials, seed
            )
            for i, hparams in enumerate(candidates, start=1):
                run_trial(
                    hparams,
                    trial_number=i,
                    trial_total=len(candidates),
                )
        else:  # "bohb"

            def objective(trial: optuna.Trial) -> float:
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
                    self._torch_trainer._cached_embedding_split(
                        prepared.train_dataloader
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
                    if space.hidden_depth is not None:
                        hparams["hidden_depth"] = trial.suggest_categorical(
                            "hidden_depth", space.hidden_depth
                        )
                    if space.hidden_size is not None:
                        hparams["hidden_size"] = trial.suggest_categorical(
                            "hidden_size", space.hidden_size
                        )
                    if space.batch_size is not None:
                        hparams["batch_size"] = trial.suggest_categorical(
                            "batch_size", space.batch_size
                        )
                if (
                    self._torch_trainer._cached_embedding_split(
                        prepared.train_dataloader
                    )
                    is None
                ):
                    [hparams] = broadcast_object_list([hparams], from_process=0)
                assert hparams is not None  # set by rank 0 above
                return run_trial(
                    hparams,
                    trial=trial if accelerator.is_main_process else None,
                    prune_enabled=True,
                    trial_number=trial.number + 1,
                    trial_total=search_config.n_trials,
                )

            study = create_optuna_study(direction, seed, search_config.n_trials)
            # Initial trials cover each width, batch and depth within n_trials.
            for hidden_size, batch_size, hidden_depth in self._probe_search_starters():
                params = {
                    "learning_rate": space.learning_rate[0],
                    "weight_decay": space.weight_decay[0],
                    "max_epochs": space.max_epochs[0],
                }
                if hidden_size is not None:
                    params["hidden_size"] = hidden_size
                if batch_size is not None:
                    params["batch_size"] = batch_size
                if hidden_depth is not None:
                    params["hidden_depth"] = hidden_depth
                study.enqueue_trial(params)
            study.optimize(objective, n_trials=search_config.n_trials)

        logger.info(
            "Best hyperparameters: %s | best %s=%.4f",
            best_hparams,
            metric_name,
            best_score,
        )
        self.selection_metric = metric_name
        self.selection_score = best_score
        assert best_hparams is not None and best_state is not None  # >= 1 trial run
        load_checkpoint_state(best_model, best_state)
        if architecture_search:
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


def fit_predictor(
    *,
    workspace: EvaluationWorkspaceConfig,
    prepared: PreparedArtifacts,
    config: TuneConfig,
    accelerator: Accelerator,
    seed: int,
    mode: Literal["tune_probe", "finetune"] = "tune_probe",
) -> FitResult:
    """Run the selected fitting protocol and return its selected model state."""
    selector = PredictorSelection(config, mode=mode)
    fitted = selector.run(
        workspace,
        prepared,
        seed=seed,
        accelerator=accelerator,
    )
    return FitResult(
        model=fitted.model,
        seed=seed,
        selected_hyperparameters=selector.selected_hyperparameters,
        layer_selections=selector.layer_selections,
        seed_variance=selector.seed_variance,
        final_fit_split=selector._final_fit_split(prepared),
        selection_metric=selector.selection_metric,
        selection_score=selector.selection_score,
    )
