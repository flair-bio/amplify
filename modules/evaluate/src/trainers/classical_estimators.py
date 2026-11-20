"""Classical probe fitting, selection and final development refits."""

from __future__ import annotations

import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from accelerate import Accelerator
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.config import EvaluationWorkspaceConfig, HeadType, TuneConfig
from modules.evaluate.src.dataset.dataloader import metric_groups_for_ids
from modules.evaluate.src.dataset.embeddings import find_embedding_dataset
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.classical_estimators import (
    build_estimator,
    classical_search_space,
    has_classical_search_space,
)
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.trainers.refit import _final_fit_split
from modules.evaluate.src.trainers.selection_helpers import (
    _log_seed_variance,
    create_optuna_study,
    resolve_search_direction,
    sample_candidates,
)
from modules.evaluate.src.trainers.torch import resolve_selection_metric

if TYPE_CHECKING:
    import optuna
logger = logging.getLogger(__name__)


def fit_candidates(
    head_type: HeadType,
    task_type: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    search_space: dict[str, list[Any]],
    default_hyperparameters: dict[str, Any],
    metric_name: str,
    direction: Literal["maximize", "minimize"],
    search_pattern: Literal["grid", "random", "bohb"] = "bohb",
    n_trials: int | None = None,
    seed: int = 710019,
    average: str = "weighted",
    num_labels: int | None = None,
    is_main_process: bool = True,
    group_ids: list[Any] | None = None,
) -> tuple[BaseEstimator, dict[str, Any], float]:
    """Fit and validate candidates, returning the training-only winner.

    Candidate selection uses only ``val_X``/``val_y``. The caller owns
    optional refitting on train+validation after selection.
    """
    effective_search_space = classical_search_space(
        head_type, task_type, search_space, default_hyperparameters
    )
    candidates: list[dict[str, Any]] = (
        list(ParameterGrid(effective_search_space))
        if effective_search_space
        else [default_hyperparameters]
    )
    candidates = [{**default_hyperparameters, **candidate} for candidate in candidates]
    if search_pattern == "random" and len(candidates) > 1:
        candidates = sample_candidates(candidates, n_trials or len(candidates), seed)
    total_trials = (
        n_trials or len(candidates)
        if search_pattern == "bohb" and effective_search_space
        else len(candidates)
    )
    if is_main_process:
        logger.info(
            "Starting classical probe search | head_type=%s | pattern=%s | trials=%d | metric=%s (%s) | train_rows=%d | validation_rows=%d",
            head_type,
            search_pattern,
            total_trials,
            metric_name,
            direction,
            len(train_y),
            len(val_y),
        )
    best_estimator: BaseEstimator | None = None
    best_hparams: dict[str, Any] | None = None
    best_score = float("-inf") if direction == "maximize" else float("inf")

    def evaluate(hparams: dict[str, Any], trial_number: int) -> float:
        nonlocal best_estimator, best_hparams, best_score
        trial_started = monotonic()
        if is_main_process:
            logger.info(
                "Classical candidate %d/%d started | head_type=%s | params=%s",
                trial_number,
                total_trials,
                head_type,
                hparams,
            )
        estimator = build_estimator(head_type, task_type, hparams, seed=seed)
        estimator.fit(train_X, train_y)
        predictions = estimator.predict(val_X)
        task = get_flat_task(task_type, num_labels)
        probabilities = (
            estimator.predict_proba(val_X)
            if hasattr(estimator, "predict_proba")
            else None
        )
        if group_ids is None:
            score = task.compute_metrics(
                predictions.tolist(),
                val_y.tolist(),
                [metric_name],
                average=average,
                probabilities=probabilities.tolist()
                if probabilities is not None
                else None,
            )[metric_name]
        else:
            score = task.compute_grouped_metrics(
                predictions.tolist(),
                val_y.tolist(),
                group_ids,
                [metric_name],
                average=average,
            )[metric_name]
        improved = score > best_score if direction == "maximize" else score < best_score
        if improved:
            best_estimator, best_hparams, best_score = (estimator, hparams, score)
        if is_main_process:
            logger.info(
                "Classical candidate %d/%d complete | val_%s=%.4f | best_%s=%.4f | new_best=%s | elapsed=%.1fs",
                trial_number,
                total_trials,
                metric_name,
                score,
                metric_name,
                best_score,
                improved,
                monotonic() - trial_started,
            )
        return score

    if search_pattern == "bohb" and effective_search_space:

        def objective(trial: optuna.Trial) -> float:
            hparams = {
                **default_hyperparameters,
                **{
                    name: trial.suggest_categorical(name, values)
                    for name, values in effective_search_space.items()
                },
            }
            return evaluate(hparams, trial.number + 1)

        trials = n_trials or len(candidates)
        study = create_optuna_study(direction, seed, trials)
        study.optimize(objective, n_trials=trials)
    else:
        for trial_number, hparams in enumerate(candidates, start=1):
            evaluate(hparams, trial_number)
    assert best_hparams is not None and best_estimator is not None
    return (best_estimator, best_hparams, best_score)


def _run_classical_head(
    config: TuneConfig,
    _fit_result: FitResult,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    seed: int,
    accelerator: Accelerator,
) -> Predictor:
    """Fit (optionally searching) a classical, gradient-free head
    directly on cached pooled embeddings, bypassing the epoch/optimizer
    loop used for ``head_type='mlp'`` entirely.

    Requires embedding dataloaders (``steps.prepare.embedding_cache.
    enabled=True``): raises if the train batch doesn't carry
    ``inputs_embeds``, since a classical estimator needs a fixed-size
    feature vector per example, not a token sequence.
    """
    if training.train_dataloader is None:
        raise ValueError("A train dataloader is required for a classical head.")
    if find_embedding_dataset(training.train_dataloader) is None:
        raise ValueError(
            f"head_type='{config.head.head_type}' requires steps.prepare.embedding_cache.enabled=True: classical heads are fit on fixed-size pooled embeddings, not token batches."
        )
    train_X, train_y = _gather_embeddings(training.train_dataloader, accelerator)
    head_config = config.head
    search_config = config.hyperparameter_search
    search_requested = (
        has_classical_search_space(
            head_config.head_type, head_config.search_space, head_config.hyperparameters
        )
        or search_config.enabled
    )
    val_X = val_y = None
    val_group_ids = None
    if training.val_dataloader is not None:
        val_X, val_y = _gather_embeddings(training.val_dataloader, accelerator)
        if training.validation_metric_groups is not None:
            validation_dataset = find_embedding_dataset(training.val_dataloader)
            if validation_dataset is None:
                raise ValueError(
                    "Grouped classical HPO requires cached validation IDs."
                )
            val_group_ids = metric_groups_for_ids(
                validation_dataset.ids, training.validation_metric_groups, "validation"
            )
    elif search_requested:
        raise ValueError(
            "Classical probe hyperparameter search requires a validation split. Add a 'validation' split or use fixed steps.tune.head.hyperparameters."
        )
    metric_name = resolve_selection_metric(config, workspace)
    direction = resolve_search_direction(
        metric_name, config.hyperparameter_search.direction
    )
    if training.val_dataloader is not None:
        best_estimator, best_hparams, best_score = fit_candidates(
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
            search_pattern=search_config.search_pattern
            if search_config.enabled
            else "grid",
            n_trials=search_config.n_trials if search_config.enabled else None,
            seed=seed,
            average=config.metric_aggregation,
            num_labels=workspace.num_labels,
            is_main_process=accelerator.is_main_process,
            group_ids=val_group_ids,
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
    if config.variance_seeds and training.val_dataloader is not None:
        scores = [
            fit_candidates(
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
                group_ids=val_group_ids,
            )[2]
            for extra_seed in config.variance_seeds
        ]
        _fit_result.seed_variance[metric_name] = scores
        _log_seed_variance(metric_name, scores)
    if config.refit_on_train_validation:
        assert val_X is not None and val_y is not None
        final_features = [train_X, val_X]
        final_labels = [train_y, val_y]
        for loader in training.refit_dataloaders.values():
            extra_X, extra_y = _gather_embeddings(loader, accelerator)
            final_features.append(extra_X)
            final_labels.append(extra_y)
        best_estimator = build_estimator(
            head_config.head_type, workspace.task_type, best_hparams, seed=seed
        )
        best_estimator.fit(np.concatenate(final_features), np.concatenate(final_labels))
        logger.info(
            "Refitted fresh %s estimator on %d rows from %s.",
            head_config.head_type,
            sum((len(labels) for labels in final_labels)),
            _final_fit_split(config, predictor, training),
        )
    predictor.model = best_estimator
    _fit_result.selected_hyperparameters = dict(best_hparams)
    _fit_result.selection_metric = metric_name
    _fit_result.selection_score = best_score
    return predictor


def _gather_embeddings(
    dataloader: Any, accelerator: Accelerator
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full (N, H)/(N,) embeddings/labels backing *dataloader*.

    *dataloader* wraps an ``EmbeddingDataset`` that's already fully
    materialized on every rank (computed once, gathered across ranks, by
    ``data preparation``/``extract_embeddings``) before ``accelerator.prepare()``
    resharded it across ranks for the (here, unused) epoch/DDP training
    loop. Reading straight off the underlying dataset avoids
    redundantly re-gathering that same data across ranks batch by batch.
    """
    dataset = find_embedding_dataset(dataloader)
    if dataset is not None:
        return (dataset.embeddings.detach().cpu().numpy(), np.asarray(dataset.labels))
    embeddings, labels = ([], [])
    for batch in dataloader:
        gathered_embeds, gathered_labels = accelerator.gather_for_metrics(
            (batch["inputs_embeds"], batch["labels"])
        )
        embeddings.append(gathered_embeds.detach().cpu().numpy())
        labels.append(gathered_labels.detach().cpu().numpy())
    return (np.concatenate(embeddings, axis=0), np.concatenate(labels, axis=0))
