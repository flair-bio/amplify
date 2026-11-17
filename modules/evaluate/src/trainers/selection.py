"""Training-only candidate search and selection."""

from __future__ import annotations

import itertools
import logging
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.config.evaluation import TuneConfig
from modules.evaluate.src.config.shared import EvaluationWorkspaceConfig
from modules.evaluate.src.dataset.types import TrainingData
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.trainers.selection_helpers import (
    checkpoint_state,
    cleanup_between_trials,
    create_optuna_study,
    load_checkpoint_state,
    sample_candidates,
)
from modules.evaluate.src.trainers.torch import (
    _cached_embedding_split,
    evaluate_torch_validation,
    resolve_selection_direction,
    resolve_selection_metric,
    train_torch,
)

if TYPE_CHECKING:
    import optuna
logger = logging.getLogger(__name__)


def _probe_search_starters(
    config: TuneConfig,
) -> list[tuple[int | None, int | None, int | None]]:
    """Initial width/batch/depth combinations covering each configured value."""
    space = config.hyperparameter_search.search_space
    if (
        space.hidden_size is None
        and space.batch_size is None
        and (space.hidden_depth is None)
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
    config: TuneConfig,
    search_pattern: Literal["bohb", "random", "grid"],
    n_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    space = config.hyperparameter_search.search_space
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
        if len(combos) > n_trials:
            logger.warning(
                "Grid search space has %d combinations, which exceeds n_trials=%d; running the full grid instead of truncating it (grid search should not subsample).",
                len(combos),
                n_trials,
            )
        candidates = combos
    else:
        if n_trials > len(combos):
            logger.warning(
                "n_trials=%d exceeds the number of unique combinations (%d); running all unique combinations once instead.",
                n_trials,
                len(combos),
            )
        if not _probe_search_starters(config):
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
                for width, batch, depth in _probe_search_starters(config)
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
    config: TuneConfig,
    train_trunk: bool,
    _fit_result: FitResult,
    workspace: EvaluationWorkspaceConfig,
    predictor: Predictor,
    training: TrainingData,
    accelerator: Accelerator,
    seed: int,
) -> dict[str, Any]:
    search_config = config.hyperparameter_search
    space = search_config.search_space
    metric_name = resolve_selection_metric(config, workspace)
    direction = resolve_selection_direction(config, metric_name)
    if accelerator.is_main_process:
        logger.info(
            "Starting hyperparameter tuning | pattern=%s | trial budget=%d | metric=%s (%s)",
            search_config.search_pattern,
            search_config.n_trials,
            metric_name,
            direction,
        )
    model = accelerator.unwrap_model(predictor.model)
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
            best_score, best_hparams = (score, hparams)
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
            predictor.model = model
        if "seed" in hparams:
            torch.manual_seed(hparams["seed"])
            train_generator = getattr(training.train_dataloader, "generator", None)
            if train_generator is not None:
                train_generator.manual_seed(hparams["seed"])
        if not architecture_search:
            load_checkpoint_state(model, initial_state)
        try:
            training_score = train_torch(
                config,
                train_trunk,
                workspace,
                predictor,
                training,
                accelerator,
                hparams,
                trial=trial,
                prune_enabled=prune_enabled,
                hpo_trial=(trial_number, trial_total),
            )
            if training_score is not None:
                score = training_score
            else:
                score = evaluate_torch_validation(
                    config,
                    workspace,
                    predictor,
                    training,
                    accelerator,
                    metric_name,
                    batch_size=hparams.get("batch_size"),
                )
        finally:
            cleanup_between_trials()
        is_best = consider(hparams, score)
        if accelerator.is_main_process:
            logger.info(
                "HPO trial %d/%d complete | val_%s=%.4f | best_%s=%.4f | new_best=%s | elapsed=%.1fs",
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
        candidates = _generate_candidates(
            config, search_config.search_pattern, search_config.n_trials, seed
        )
        for i, hparams in enumerate(candidates, start=1):
            run_trial(hparams, trial_number=i, trial_total=len(candidates))
    else:

        def objective(trial: optuna.Trial) -> float:
            if accelerator.is_main_process:
                hparams = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        min(space.learning_rate),
                        max(space.learning_rate),
                        log=True,
                    ),
                    "weight_decay": _suggest_weight_decay(config, trial),
                    "max_epochs": trial.suggest_categorical(
                        "max_epochs", space.max_epochs
                    ),
                    "seed": seed,
                }
            elif _cached_embedding_split(training.train_dataloader) is not None:
                hparams = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        min(space.learning_rate),
                        max(space.learning_rate),
                        log=True,
                    ),
                    "weight_decay": _suggest_weight_decay(config, trial),
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
            if _cached_embedding_split(training.train_dataloader) is None:
                [hparams] = broadcast_object_list([hparams], from_process=0)
            assert hparams is not None
            return run_trial(
                hparams,
                trial=trial if accelerator.is_main_process else None,
                prune_enabled=True,
                trial_number=trial.number + 1,
                trial_total=search_config.n_trials,
            )

        study = create_optuna_study(direction, seed, search_config.n_trials)
        for hidden_size, batch_size, hidden_depth in _probe_search_starters(config):
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
        "Best hyperparameters: %s | best %s=%.4f", best_hparams, metric_name, best_score
    )
    _fit_result.selection_metric = metric_name
    _fit_result.selection_score = best_score
    assert best_hparams is not None and best_state is not None
    load_checkpoint_state(best_model, best_state)
    if architecture_search:
        predictor.model = best_model
    return best_hparams


def _suggest_weight_decay(config: TuneConfig, trial: Any) -> float:
    space = config.hyperparameter_search.search_space
    if space.weight_decay_sampling == "log":
        return trial.suggest_float(
            "weight_decay", min(space.weight_decay), max(space.weight_decay), log=True
        )
    return trial.suggest_categorical("weight_decay", space.weight_decay)
