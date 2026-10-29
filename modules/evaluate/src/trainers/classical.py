"""Training and validation for classical embedding probes."""

from __future__ import annotations

import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from modules.evaluate.src.contracts.config import HeadType
from modules.evaluate.src.model import heads
from modules.evaluate.src.model.heads import classical_search_space
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.trainers.selection_helpers import (
    create_optuna_study,
    sample_candidates,
)

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


class ClassicalProbeTrainer:
    """Fit candidate estimators and select by their validation metric."""

    @staticmethod
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
    ) -> tuple[BaseEstimator, dict[str, Any], float]:
        """Fit and validate candidates, returning the training-only winner.

        Candidate selection uses only ``val_X``/``val_y``. The caller owns
        optional refitting on train+validation after selection.
        """
        effective_search_space = classical_search_space(
            head_type,
            task_type,
            search_space,
            default_hyperparameters,
        )
        candidates: list[dict[str, Any]] = (
            list(ParameterGrid(effective_search_space))
            if effective_search_space
            else [default_hyperparameters]
        )
        candidates = [
            {**default_hyperparameters, **candidate} for candidate in candidates
        ]

        if search_pattern == "random" and len(candidates) > 1:
            candidates = sample_candidates(
                candidates, n_trials or len(candidates), seed
            )

        total_trials = (
            n_trials or len(candidates)
            if search_pattern == "bohb" and effective_search_space
            else len(candidates)
        )
        if is_main_process:
            logger.info(
                "Starting classical probe search | head_type=%s | pattern=%s | "
                "trials=%d | metric=%s (%s) | train_rows=%d | validation_rows=%d",
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
            estimator = heads.build_estimator(head_type, task_type, hparams, seed=seed)
            estimator.fit(train_X, train_y)
            predictions = estimator.predict(val_X)
            task = get_flat_task(task_type, num_labels)
            probabilities = (
                estimator.predict_proba(val_X)
                if hasattr(estimator, "predict_proba")
                else None
            )
            score = task.compute_metrics(
                predictions.tolist(),
                val_y.tolist(),
                [metric_name],
                average=average,
                probabilities=(
                    probabilities.tolist() if probabilities is not None else None
                ),
            )[metric_name]
            improved = (
                score > best_score if direction == "maximize" else score < best_score
            )
            if improved:
                best_estimator, best_hparams, best_score = estimator, hparams, score
            if is_main_process:
                logger.info(
                    "Classical candidate %d/%d complete | val_%s=%.4f | "
                    "best_%s=%.4f | new_best=%s | elapsed=%.1fs",
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
        return best_estimator, best_hparams, best_score
