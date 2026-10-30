"""Hyperparameter search loop for cached MLP probes (extracted from TuneStep).

Random/grid searches enumerate candidates with ``training.search``; TPE uses
Optuna with median pruning. The trainer supplies the training loop and
validation scoring, so this module owns only trial bookkeeping.
"""

from __future__ import annotations

import logging
from time import monotonic
from typing import TYPE_CHECKING, Any

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.training.search import (
    generate_candidates,
    probe_search_starters,
    suggest_weight_decay,
)
from modules.evaluate.src.utils.hpo import (
    checkpoint_state,
    cleanup_between_trials,
    create_optuna_study,
    load_checkpoint_state,
)

if TYPE_CHECKING:
    import optuna

    from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer

logger = logging.getLogger(__name__)


def search_hyperparameters(
    trainer: TorchProbeTrainer,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Accelerator,
    seed: int,
) -> dict[str, Any]:
    """Run the configured search and leave the best model in ``prepared``."""
    search_config = trainer.config.hyperparameter_search
    space = search_config.search_space
    metric_name = trainer.resolve_metric_name(workspace)
    direction = trainer.resolve_direction(metric_name)
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
            training_score = trainer.train(
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
                score = trainer.evaluate_validation(
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

    if (
        search_config.search_pattern == "random"
        or search_config.search_pattern == "grid"
    ):
        candidates = generate_candidates(
            space,
            search_config.search_pattern,
            search_config.n_trials,
            seed,
        )
        for i, hparams in enumerate(candidates, start=1):
            run_trial(
                hparams,
                trial_number=i,
                trial_total=len(candidates),
            )
    else:  # "tpe" (legacy "bohb" is normalized to "tpe" by the config)

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
                    # fixed categorical set: TPE can only exploit the shape
                    # of the search space for continuous parameters like
                    # learning_rate if it's allowed to draw values between
                    # the configured bounds, not just snap to a handful of
                    # discrete choices.
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        min(space.learning_rate),
                        max(space.learning_rate),
                        log=True,
                    ),
                    "weight_decay": suggest_weight_decay(trial, space),
                    "max_epochs": trial.suggest_categorical(
                        "max_epochs", space.max_epochs
                    ),
                    "seed": seed,
                }
            elif trainer.cached_embedding_split(prepared.train_dataloader) is not None:
                hparams = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate",
                        min(space.learning_rate),
                        max(space.learning_rate),
                        log=True,
                    ),
                    "weight_decay": suggest_weight_decay(trial, space),
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
            if trainer.cached_embedding_split(prepared.train_dataloader) is None:
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
        for hidden_size, batch_size, hidden_depth in probe_search_starters(space):
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
    assert best_hparams is not None and best_state is not None  # >= 1 trial run
    load_checkpoint_state(best_model, best_state)
    if architecture_search:
        prepared.model = best_model
    return best_hparams
