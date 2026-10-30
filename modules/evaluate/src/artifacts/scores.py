"""Score artifacts: persist computed scores, bootstrap and layerwise results.

Computation lives in :mod:`modules.evaluate.src.scoring.compute`; this module
only writes parquet files and run-manifest sections, and upserts long-format
rows. It never recomputes a metric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.schemas.artifacts import ContactScoreOutput
from modules.evaluate.src.scoring.compute import BootstrapResult
from modules.evaluate.src.utils.io import (
    update_run_manifest,
    write_parquet,
    write_settings_json,
)
from modules.evaluate.src.utils.long_format import upsert_long_df

LAYERWISE_COMPARISON_FILENAME = "layerwise_comparison.parquet"


def publish_contact_scores(
    result: ContactScoreOutput, output_dir: Path, config: dict[str, Any]
) -> None:
    """Write CASP unit/summary scores, control scores, and the settings file.

    Sets the ``*_path`` fields on *result* in place.
    """
    seed = result.metadata.seed
    assert seed is not None
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
            result.control_summaries[name],
            control_dir / "scores.parquet",
        )
    write_settings_json(
        output_dir / "contact_score.json",
        seed,
        config,
        metadata=result.metadata.model_dump(mode="json"),
    )


def publish_condition_scores(
    workspace: EvaluationWorkspace,
    condition_scores: dict[str, dict[str, float]],
    metadata_dict: dict[str, Any],
) -> Path:
    """Write one row per (condition, metric) to ``condition_scores.parquet``."""
    return write_parquet(
        pl.DataFrame(
            [
                {
                    **metadata_dict,
                    "condition": condition,
                    "metric": metric,
                    "score": value,
                }
                for condition, metrics in condition_scores.items()
                for metric, value in metrics.items()
            ]
        ),
        workspace.run_root / "condition_scores.parquet",
    )


def publish_score_manifest(
    workspace: EvaluationWorkspace,
    payload: dict[str, Any],
) -> Path:
    """Record the point-estimate scores in the run manifest's ``score`` section."""
    return update_run_manifest(workspace.run_manifest_path, "score", payload)


def publish_bootstrap(
    workspace: EvaluationWorkspace,
    result: BootstrapResult,
    *,
    output_filename: str,
    seed: int,
    config: dict[str, Any] | None,
    metadata_dict: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Write the resample table and the ``bootstrap`` manifest section.

    Returns ``(resamples_path, manifest_path)``.
    """
    output_path = write_parquet(
        result.resamples, workspace.stats_path / output_filename
    )
    summary_path = update_run_manifest(
        workspace.run_manifest_path,
        "bootstrap",
        {
            "seed": seed,
            "config": config,
            "metadata": metadata_dict,
            "confidence_intervals": result.confidence_intervals,
            "control_comparisons": result.control_comparisons,
        },
    )
    return output_path, summary_path


def publish_long_scores(
    long_path: Path,
    long_df: pl.DataFrame,
    variant: str | None,
    split: str | None,
) -> tuple[pl.DataFrame, Path]:
    """Upsert this run's rows into the tidy long-format sidecar and write it.

    Returns ``(combined_frame, path)``.
    """
    combined = upsert_long_df(long_path, long_df, variant, split)
    return combined, write_parquet(combined, long_path)


def publish_layerwise(
    workspace: EvaluationWorkspace,
    comparison: pl.DataFrame,
    *,
    split: str | None,
) -> Path:
    """Upsert cross-layer comparison rows into ``layerwise_comparison.parquet``."""
    path = workspace.comparison_path / LAYERWISE_COMPARISON_FILENAME
    return write_parquet(
        upsert_long_df(path, comparison, workspace.variant, split),
        path,
    )
