"""Classical (gradient-free) head fitting on cached pooled embeddings.

Extracted from the legacy ``TuneStep._run_classical_head``. Fits a sklearn-style
estimator, optionally sweeping hyperparameters on validation data, instead of
the epoch/optimizer loop used by ``head_type='mlp'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import joblib
import numpy as np
from accelerate import Accelerator

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import (
    build_estimator,
    has_classical_search_space,
    sweep_classical_head,
)
from modules.evaluate.src.predictors.embeddings import find_embedding_dataset
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts

if TYPE_CHECKING:
    from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer

logger = logging.getLogger(__name__)


def run_classical_head(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    seed: int,
    accelerator: Accelerator,
) -> PreparedArtifacts:
    """Fit (optionally searching) a classical head on cached pooled embeddings.

    Requires embedding dataloaders (``steps.prepare.embedding_cache.enabled=True``):
    raises if the train batch doesn't carry ``inputs_embeds``, since a classical
    estimator needs a fixed-size feature vector per example, not a token sequence.
    """
    config = trainer.config
    if prepared.train_dataloader is None:
        raise ValueError("A train dataloader is required for a classical head.")
    # Checked via the underlying dataset rather than next(iter(...)):
    # iterating the accelerator-prepared dataloader here would trigger
    # Accelerate's DataLoaderShard RNG resync collective on its first
    # ever iteration for this run, which other ranks (skipping straight
    # to TorchProbeTrainer.gather_embeddings's dataset-based read) never join.
    if find_embedding_dataset(prepared.train_dataloader) is None:
        raise ValueError(
            f"head_type='{config.head.head_type}' requires "
            "steps.prepare.embedding_cache.enabled=True: classical heads "
            "are fit on fixed-size pooled embeddings, not token batches."
        )

    train_X, train_y = trainer.gather_embeddings(prepared.train_dataloader, accelerator)
    head_config = config.head
    search_config = config.hyperparameter_search
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
        val_X, val_y = trainer.gather_embeddings(prepared.val_dataloader, accelerator)
    elif search_requested:
        raise ValueError(
            "Classical probe hyperparameter search requires a validation "
            "split. Add a 'validation' split or use fixed "
            "steps.tune.head.hyperparameters."
        )

    metric_name = trainer.resolve_metric_name(workspace)
    direction = trainer.resolve_direction(metric_name)

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
                search_config.engine_pattern if search_config.enabled else "grid"
            ),
            n_trials=search_config.n_trials if search_config.enabled else None,
            seed=seed,
            average=config.metric_aggregation,
            num_labels=workspace.num_labels,
            is_main_process=accelerator.is_main_process,
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
        logger.info(
            "Fitted head_type=%s with fixed hparams=%s on train only (no validation split)",
            head_config.head_type,
            best_hparams,
        )

    if config.variance_seeds and prepared.val_dataloader is not None:
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
                average=config.metric_aggregation,
                num_labels=workspace.num_labels,
                is_main_process=accelerator.is_main_process,
            )[2]
            for extra_seed in config.variance_seeds
        ]
        trainer.log_seed_variance(metric_name, scores)

    if config.refit_on_train_validation:
        assert val_X is not None and val_y is not None
        final_features = [train_X, val_X]
        final_labels = [train_y, val_y]
        for loader in prepared.refit_dataloaders.values():
            extra_X, extra_y = trainer.gather_embeddings(loader, accelerator)
            final_features.append(extra_X)
            final_labels.append(extra_y)
        best_estimator = build_estimator(
            head_config.head_type, workspace.task_type, best_hparams, seed=seed
        )
        best_estimator.fit(np.concatenate(final_features), np.concatenate(final_labels))
        logger.info(
            "Refitted fresh %s estimator on %d rows from %s.",
            head_config.head_type,
            sum(len(labels) for labels in final_labels),
            trainer.final_fit_split(prepared),
        )
    prepared.model = best_estimator

    if config.save_best_model and accelerator.is_main_process:
        # Every rank fit the same estimator on the same gathered data and
        # seed, so only the main process needs to write it out.
        workspace.model_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_estimator, workspace.model_path / "head.joblib")
        trainer.write_tune_settings(
            workspace.model_path / "tune_config.json",
            seed,
            best_hparams,
            final_fit_split=trainer.final_fit_split(prepared),
        )
        logger.info(
            "Saved best %s head to %s", head_config.head_type, workspace.model_path
        )
    accelerator.wait_for_everyone()

    return prepared
