"""Generic evaluation lifecycle engine.

One run = reserve outputs, prepare observations, fit one predictor, predict and
score the main view, then predict and score each configured external view with
the same fitted state. The engine branches only on capabilities, never on task
names, and holds no metric, training, or artifact logic of its own.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_external_workspace,
)
from modules.evaluate.src.evaluation.contracts import (
    FitSummary,
    PredictionBundle,
    RunContext,
    ScoringResult,
    TaskDefinition,
)
from modules.evaluate.src.evaluation.reservation import reserve_outputs

logger = logging.getLogger(__name__)


class SingleProcessRequired(ValueError):
    """Raised when a single-process task is launched with several processes."""


@dataclass(frozen=True)
class ViewOutcome:
    """Predictions for one evaluation view and their scores (main process only).

    ``scores`` holds one result per prediction branch, in branch order; it is
    ``None`` on processes that do not score.
    """

    predictions: tuple[PredictionBundle, ...]
    scores: tuple[ScoringResult, ...] | None


@dataclass(frozen=True)
class EvaluationOutcome:
    """Everything the generic lifecycle produced for one evaluation run."""

    fit: FitSummary
    main: ViewOutcome
    external: dict[str, ViewOutcome] = field(default_factory=dict)


def run_evaluation(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    *,
    task: TaskDefinition,
    dataset: Any = None,
    overwrite: bool = False,
) -> EvaluationOutcome:
    """Run one task end to end inside its output reservation."""
    if task.requires_single_process and accelerator.num_processes != 1:
        raise SingleProcessRequired(
            f"{task.name} jobs are single-process; relaunch without "
            "multi-process accelerate settings."
        )
    context = RunContext(
        config=config, workspace=workspace, accelerator=accelerator, dataset=dataset
    )
    with reserve_outputs(config, workspace, accelerator, overwrite=overwrite):
        logger.info("=== Starting Prepare Step ===")
        prepared = task.prepare(context)
        predictor = task.predictor(context)
        logger.info("=== Fitting predictor %s ===", predictor.name)
        fit = predictor.fit(context, prepared)
        main = _evaluate_view(task, predictor, context, fit.fitted)

        external: dict[str, ViewOutcome] = {}
        if config.external_test_sets and not predictor.transfers_to_external:
            logger.warning(
                "Skipping external_test_sets: predictor '%s' does not transfer "
                "to external views.",
                predictor.name,
            )
        for external_set in (
            config.external_test_sets if predictor.transfers_to_external else ()
        ):
            logger.info("=== Starting external test set '%s' ===", external_set.name)
            external_workspace = get_external_workspace(workspace, external_set)
            external_workspace.setup_directories()
            external_context = replace(
                context, workspace=external_workspace, split=external_set.name
            )
            external_prepared = task.prepare_external(context, fit.fitted, external_set)
            external[external_set.name] = _evaluate_view(
                task, predictor, external_context, external_prepared
            )
            logger.info("=== Completed external test set '%s' ===", external_set.name)
        accelerator.wait_for_everyone()
        return EvaluationOutcome(fit=fit.summary, main=main, external=external)


def _evaluate_view(
    task: TaskDefinition, predictor: Any, context: RunContext, prepared: Any
) -> ViewOutcome:
    """Predict on every process; score each branch and finalize on the main process."""
    predictions = tuple(predictor.predict(context, prepared))
    scores = None
    if context.accelerator.is_main_process:
        scores = tuple(task.score(context, bundle) for bundle in predictions)
        task.finalize_view(context, predictions, scores)
    return ViewOutcome(predictions=predictions, scores=scores)
