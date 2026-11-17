"""Evaluation report publication, separate from scientific scoring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.artifacts.files import write_parquet
from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.results import (
    PredictOutput,
    ScoreOutput,
    StructuredScoreOutput,
)
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config.evaluation import ScoreConfig
from modules.evaluate.src.metrics.scoring import compare_layer_scores
from modules.evaluate.src.metrics.tables import merge_long_df
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.utils.tracking import log_wandb

logger = logging.getLogger(__name__)
LAYERWISE_COMPARISON_FILENAME = "layerwise_comparison.parquet"


def publish_scores(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    result: ScoreOutput,
    baseline_references: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> ScoreOutput:
    metadata_dict = result.metadata.model_dump(mode="json") if result.metadata else {}
    layer_index = result.metadata.layer_index if result.metadata else None
    wandb_prefix = "test" if layer_index is None else f"test/layer_{layer_index}"
    log_wandb(
        {f"{wandb_prefix}/{name}": value for name, value in result.scores.items()}
    )
    for name, reference in baseline_references.items():
        reference["scores"] = result.condition_scores[name]
    result.scores_path = update_run_manifest(
        workspace.run_manifest_path,
        "score",
        {
            "metadata": metadata_dict,
            "config": result.config,
            "scores": result.scores,
            "baselines": baseline_references,
            "condition_scores": result.condition_scores,
        },
        namespace=workspace.manifest_namespace,
    )
    if result.resamples is not None:
        result.summary_path = update_run_manifest(
            workspace.run_manifest_path,
            "bootstrap",
            {
                "seed": seed,
                "config": result.config,
                "metadata": metadata_dict,
                "confidence_intervals": result.confidence_intervals,
                "control_comparisons": result.control_comparisons or {},
            },
            namespace=workspace.manifest_namespace,
        )
        logger.info("Bootstrap summary written to %s", result.summary_path)
    long_path = workspace.comparison_path / config.long_filename
    variant = result.metadata.variant if result.metadata else None
    split = result.metadata.split if result.metadata else None
    assert result.long_df is not None
    result.long_df = merge_long_df(
        pl.read_parquet(long_path) if long_path.exists() else None,
        result.long_df,
        variant,
        split,
    )
    result.long_path = write_parquet(result.long_df, long_path)
    return result


def publish_layer_comparison(
    config: ScoreConfig,
    workspace: EvaluationWorkspace,
    layer_predictions: dict[int, PredictOutput],
    seed: int = 710019,
) -> Path | None:
    """Publish paired comparisons with the preselected deepest layer."""
    comparison = compare_layer_scores(
        get_flat_task(workspace.task_type, workspace.num_labels),
        config,
        layer_predictions,
        variant=workspace.variant,
        seed=seed,
    )
    if comparison is None:
        return None
    path = workspace.comparison_path / LAYERWISE_COMPARISON_FILENAME
    return write_parquet(
        merge_long_df(
            pl.read_parquet(path) if path.exists() else None,
            comparison,
            workspace.variant,
            layer_predictions[max(layer_predictions)].metadata.split,
        ),
        path,
    )


def publish_structured_scores(
    config: ScoreConfig,
    *,
    result: StructuredScoreOutput,
    output_dir: Path,
    manifest_path: Path,
    baseline_references: dict[str, dict[str, Any]],
) -> StructuredScoreOutput:
    logger.info("Scores: %s", result.summaries)
    seed = result.metadata.seed
    if seed is None:
        raise ValueError("Scoring requires an explicit run seed")
    result.unit_scores_path = write_parquet(
        result.unit_scores, output_dir / "unit_scores.parquet"
    )
    result.summaries_path = write_parquet(
        result.summaries, output_dir / "scores.parquet"
    )
    for name, frame in result.control_unit_scores.items():
        control_dir = output_dir / "controls" / name
        result.control_unit_scores_paths[name] = write_parquet(
            frame, control_dir / "unit_scores.parquet"
        )
        result.control_summaries_paths[name] = write_parquet(
            result.control_summaries[name], control_dir / "scores.parquet"
        )
    for name, reference in baseline_references.items():
        if name in result.control_summaries:
            reference["scores"] = result.control_summaries[name].to_dicts()
            reference["scores_path"] = str(result.control_summaries_paths[name])
            reference["unit_scores_path"] = str(result.control_unit_scores_paths[name])
    update_run_manifest(
        manifest_path,
        "score",
        {
            "seed": seed,
            "metadata": result.metadata.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "scores": result.summaries.to_dicts(),
            "scores_path": str(result.summaries_path),
            "unit_scores_path": str(result.unit_scores_path),
            "controls": {
                name: {
                    "scores": frame.to_dicts(),
                    "scores_path": str(result.control_summaries_paths[name]),
                    "unit_scores_path": str(result.control_unit_scores_paths[name]),
                }
                for name, frame in result.control_summaries.items()
            },
            "baselines": baseline_references,
        },
        namespace=("tests", result.metadata.test_subset)
        if result.metadata.test_subset
        else (),
    )
    return result
