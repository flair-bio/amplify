"""Prepare -> Tune -> Predict -> Score for one evaluation run, shared by
ordinary evaluation and each cross-validation fold."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from datasets import DatasetDict

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.schemas.artifacts import (
    CrossValidationIdentity,
    PredictOutput,
    ScoreOutput,
)
from modules.evaluate.src.steps.predict import PredictStep
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.steps.tune import TuneStep

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outputs keyed by layer index (``None`` for ordinary single-head runs).
    ``scores`` is populated on the main process only."""

    predictions: dict[int | None, PredictOutput]
    scores: dict[int | None, ScoreOutput] = field(default_factory=dict)


def run_pipeline(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    dataset: DatasetDict | None = None,
    cross_validation: CrossValidationIdentity | None = None,
) -> PipelineResult:
    """Run every step once for *workspace*, optionally on a supplied dataset."""
    logger.info("=== Starting Prepare Step ===")
    prepared_artifacts = PrepareStep(config.steps.prepare).run(
        workspace,
        seed=config.seed,
        accelerator=accelerator,
        layerwise_config=config.steps.layerwise,
        dataset=dataset,
        cross_validation=cross_validation,
    )
    logger.info("=== Completed Prepare Step ===")

    # Train in tune_probe or finetune mode. evaluate_as_is evaluates the task
    # head loaded by PrepareStep without entering TuneStep.
    mode = config.steps.mode
    tuned = mode != "evaluate_as_is"
    if mode != "evaluate_as_is":
        logger.info("=== Fine-tuning model ===")
        prepared_artifacts = TuneStep(config.steps.tune, mode=mode).run(
            workspace,
            prepared_artifacts,
            seed=config.seed,
            accelerator=accelerator,
            controls=config.steps.predict.controls,
        )
        logger.info("=== Completed Fine-tune Step ===")
    else:
        # evaluate_as_is returns a raw model from PrepareStep, so it still
        # needs accelerator.prepare() before Predict/Score can use it.
        prepared_artifacts.model = accelerator.prepare(prepared_artifacts.model)

    logger.info("=== Starting Predict Step ===")
    predict_step = PredictStep(config.steps.predict)
    layer_predictions: dict[int, PredictOutput] = {}
    predictions: dict[int | None, PredictOutput]
    if config.steps.layerwise.enabled:
        layer_predictions = predict_step.run_layerwise(
            workspace, prepared_artifacts, accelerator, config.seed
        )
        predictions = dict(layer_predictions.items())
    else:
        predictions = {
            None: predict_step.run(
                workspace,
                prepared_artifacts,
                seed=config.seed,
                accelerator=accelerator,
                # head_type doesn't apply to the "pretrained" (steps.tune never
                # ran) condition -- there's no trained head to name.
                head_type=config.steps.tune.head.head_type if tuned else None,
                variant=workspace.variant,
            )
        }
    logger.info("=== Completed Predict Step ===")

    result = PipelineResult(predictions)
    # ScoreStep is a plain, single-process step, so only rank 0 runs it.
    # Passing PredictOutput avoids a disk round trip; ScoreStep still supports
    # manifest-based reloading for standalone or restarted scoring.
    if accelerator.is_main_process:
        logger.info("=== Starting Score Step ===")
        score_step = ScoreStep(config.steps.score)
        for layer, prediction in predictions.items():
            # Separate layer workspaces keep per-depth scores and manifests apart.
            score_workspace = (
                workspace
                if layer is None
                else get_evaluation_workspace(
                    config.workspace, variant=f"{workspace.variant}-layer_{layer}"
                )
            )
            score = score_step.run(
                score_workspace, seed=config.seed, predict_result=prediction
            )
            result.scores[layer] = score
            label = "Scores" if layer is None else f"Layer {layer} scores"
            logger.info("%s: %s", label, score.scores)
            if score.summary_path is not None:
                logger.info(
                    "Bootstrap confidence intervals written to %s", score.summary_path
                )
        if layer_predictions and config.steps.score.bootstrap_enabled:
            comparison_path = score_step.compare_layers(
                workspace, layer_predictions, seed=config.seed
            )
            if comparison_path is not None:
                logger.info(
                    "Paired cross-layer comparison written to %s", comparison_path
                )
        logger.info("=== Completed Score Step ===")
    accelerator.wait_for_everyone()
    return result
