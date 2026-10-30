"""Scoring orchestration: metrics, bootstrap, layer comparison and publication.

Pure computation lives in :mod:`modules.evaluate.src.scoring.compute`; file and
manifest writing lives in :mod:`modules.evaluate.src.artifacts.scores`. This
module only sequences them for one prediction result, taking the task handler
from the generic registry rather than naming concrete tasks.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from modules.evaluate.src.artifacts.baselines import attach_baselines
from modules.evaluate.src.artifacts.predictions import (
    load_predict_output,
    output_workspace,
    record_token_positions,
)
from modules.evaluate.src.artifacts.scores import (
    publish_bootstrap,
    publish_condition_scores,
    publish_layerwise,
    publish_long_scores,
    publish_score_manifest,
)
from modules.evaluate.src.config.score import ScoreConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.evaluation.contracts import PredictionBundle, ScoringResult
from modules.evaluate.src.schemas.artifacts import PredictOutput, ScoreOutput
from modules.evaluate.src.scoring.compute import (
    bootstrap_scores,
    layerwise_comparison,
    score_conditions,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.base import TaskHandler
from modules.evaluate.src.utils.baselines import identity_hash
from modules.evaluate.src.utils.long_format import build_long_df
from modules.evaluate.src.utils.tracking import log_wandb

logger = logging.getLogger(__name__)


def score_bundle(
    config: ScoreConfig, bundle: PredictionBundle, *, seed: int
) -> ScoringResult:
    """Score one prediction branch of a generic-runner view and log the outcome."""
    score = score_predictions(
        config, bundle.workspace, seed=seed, predict_result=bundle.payload
    )
    logger.info("Scores: %s", score.scores)
    if score.summary_path is not None:
        logger.info("Bootstrap confidence intervals written to %s", score.summary_path)
    return ScoringResult(
        predictor=bundle.predictor,
        split=bundle.split,
        branch=bundle.branch,
        payload=score,
    )


def score_predictions(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    seed: int = 710019,
    predict_result: PredictOutput | None = None,
) -> ScoreOutput:
    """Score the trained model's predictions for *workspace*.

    When *predict_result* is provided, scoring uses that already-gathered
    in-memory result and avoids reloading prediction/control Parquet files.
    Otherwise it reconstructs the result from the run manifest for standalone
    or restartable scoring.
    """
    if predict_result is None:
        predict_result = load_predict_output(workspace)
    if predict_result.metadata.task_type != workspace.task_type:
        raise ValueError("Prediction task does not match the scoring workspace.")
    return _score_predictions(config, workspace, seed, predict_result, {})


def compare_predictions(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    *,
    baselines: list[str],
    controls: list[str] | None = None,
    seed: int = 710019,
    predict_result: PredictOutput | None = None,
) -> ScoreOutput:
    """Compare only explicitly selected saved conditions, without fitting or inference.

    Outputs are separate from standalone model and baseline score artifacts.
    Historical ``baseline_context.names`` is deliberately not consulted.
    """
    if predict_result is None:
        predict_result = load_predict_output(workspace)
    if predict_result.metadata.task_type != workspace.task_type:
        raise ValueError("Prediction task does not match the scoring workspace.")
    if (
        predict_result.baseline_context is not None
        and predict_result.baseline_context["fit"]["num_labels"] != workspace.num_labels
    ):
        raise ValueError(
            "Prediction class contract does not match the scoring workspace."
        )
    controls = controls or []
    if not baselines and not controls:
        raise ValueError("Select at least one baseline or model control to compare.")
    available = set(predict_result.controls) | set(predict_result.requested_controls)
    missing = set(controls) - available
    if missing:
        raise ValueError(
            f"Missing saved model controls: {sorted(missing)}. Rerun comparison with existing controls."
        )
    if len(set(controls)) != len(controls):
        raise ValueError("Comparison controls must be unique.")
    selected_controls = {}
    for name in controls:
        if name == "scrambled_labels":
            continue
        if name not in predict_result.controls:
            raise ValueError(f"Missing saved predictions for control '{name}'.")
        frame = predict_result.controls[name]
        model_ids = predict_result.predictions["id"].to_list()
        control_ids = frame["id"].to_list()
        if len(control_ids) != len(model_ids):
            raise ValueError(f"Control '{name}' has a different evaluated population.")
        if control_ids != model_ids:
            if len(set(model_ids)) != len(model_ids) or len(set(control_ids)) != len(
                control_ids
            ):
                raise ValueError(
                    f"Control '{name}' has ambiguous repeated example IDs."
                )
            if set(control_ids) != set(model_ids):
                raise ValueError(
                    f"Control '{name}' IDs do not match model predictions."
                )
            indices = {
                identifier: index for index, identifier in enumerate(control_ids)
            }
            frame = frame[[indices[identifier] for identifier in model_ids]]
        if workspace.task_type == "token_classification":
            if (
                "evaluated_positions" not in frame.columns
                or "evaluated_positions" not in predict_result.predictions.columns
            ):
                raise ValueError(
                    f"Token control '{name}' lacks evaluated residue positions/masks; recover original coordinates before comparison."
                )
            if (
                frame["evaluated_positions"].to_list()
                != predict_result.predictions["evaluated_positions"].to_list()
            ):
                raise ValueError(
                    f"Token control '{name}' residue positions/masks do not match."
                )
        selected_controls[name] = frame
    selected = replace(
        predict_result,
        predictions=record_token_positions(
            predict_result.predictions, predict_result.baseline_context
        ),
        controls=selected_controls,
        requested_controls=list(controls),
    )
    task = get_task(workspace.task_type, workspace.num_labels)
    selected, references = attach_baselines(
        selected, regression=task.is_regression, names=baselines
    )
    protocol = {
        "baselines": baselines,
        "controls": controls,
        "config": config.model_dump(mode="json"),
        "seed": seed,
    }
    comparison_workspace = output_workspace(
        workspace, workspace.run_root / "comparisons" / identity_hash(protocol)[:16]
    )
    return _score_predictions(config, comparison_workspace, seed, selected, references)


def _score_predictions(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    seed: int,
    predict_result: PredictOutput,
    baseline_references: dict[str, dict[str, Any]],
) -> ScoreOutput:
    logger.info("Running Score step...")
    if predict_result.predictions.is_empty():
        raise ValueError(
            "score_predictions requires PredictOutput.predictions to be "
            "populated (run prediction before scoring)."
        )

    task = get_task(workspace.task_type, workspace.num_labels)
    metric_names = config.metrics or task.default_metrics()
    logger.info("Computing metrics: %s", metric_names)
    scored = score_conditions(
        task,
        predict_result,
        metric_names,
        config.metric_aggregation,
        seed,
    )
    scores = scored.trained

    config_dict = config.model_dump(mode="json")
    layer_index = predict_result.metadata.layer_index
    wandb_prefix = "test" if layer_index is None else f"test/layer_{layer_index}"
    log_wandb({f"{wandb_prefix}/{name}": value for name, value in scores.items()})
    metadata_dict = predict_result.metadata.model_dump(mode="json")
    for name, reference in baseline_references.items():
        reference["scores"] = scored.conditions[name]
    condition_scores_path = publish_condition_scores(
        workspace, scored.conditions, metadata_dict
    )
    scores_path = publish_score_manifest(
        workspace,
        {
            "metadata": metadata_dict,
            "config": config_dict,
            "scores": scores,
            "baselines": baseline_references,
            "condition_scores_path": str(condition_scores_path),
        },
    )

    result = ScoreOutput(
        scores=scores,
        metadata=predict_result.metadata,
        scores_path=scores_path,
        config=config_dict,
    )

    if config.bootstrap_enabled:
        run_bootstrap(
            config,
            workspace,
            task,
            predict_result,
            metric_names,
            seed,
            result,
            metadata_dict,
            trained_per_example_scores=scored.trained_per_example,
        )

    long_df = build_long_df(
        result.metadata,
        result.scores,
        result.confidence_intervals,
        result.control_comparisons,
    )
    variant = result.metadata.variant if result.metadata is not None else None
    split = result.metadata.split if result.metadata is not None else None
    result.long_df, result.long_path = publish_long_scores(
        workspace.comparison_path / config.long_filename,
        long_df,
        variant,
        split,
    )
    return result


def compare_layers(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    layer_predictions: dict[int, PredictOutput],
    seed: int = 710019,
) -> Path | None:
    """Compare each probed layer with the deepest one on the same test rows.

    Returns None when fewer than two layers were probed. See
    :func:`~modules.evaluate.src.scoring.compute.layerwise_comparison` for
    the paired-bootstrap and permutation semantics.
    """
    if len(layer_predictions) < 2:
        return None
    task = get_task(workspace.task_type, workspace.num_labels)
    metric_names = config.metrics or task.default_metrics()
    comparison = layerwise_comparison(
        task,
        layer_predictions,
        metric_names,
        config.metric_aggregation,
        variant=workspace.variant,
        n_samples=config.n_samples,
        confidence_interval=config.confidence_interval,
        n_permutations=config.n_permutations,
        seed=seed,
    )
    reference = max(layer_predictions)
    return publish_layerwise(
        workspace,
        comparison,
        split=layer_predictions[reference].metadata.split,
    )


def run_bootstrap(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    task: TaskHandler,
    predict_result: PredictOutput,
    metric_names: list[str],
    seed: int,
    result: ScoreOutput,
    metadata_dict: dict | None,
    *,
    trained_per_example_scores: np.ndarray | None,
) -> None:
    bootstrap = bootstrap_scores(
        task,
        predict_result,
        metric_names,
        config.metric_aggregation,
        n_samples=config.n_samples,
        confidence_interval=config.confidence_interval,
        n_permutations=config.n_permutations,
        seed=seed,
        trained_per_example_scores=trained_per_example_scores,
    )
    output_path, summary_path = publish_bootstrap(
        workspace,
        bootstrap,
        output_filename=config.output_filename,
        seed=seed,
        config=result.config,
        metadata_dict=metadata_dict,
    )
    result.confidence_intervals = bootstrap.confidence_intervals
    result.output_path = output_path
    result.resamples = bootstrap.resamples
    result.summary_path = summary_path
    result.control_comparisons = bootstrap.control_comparisons or None
