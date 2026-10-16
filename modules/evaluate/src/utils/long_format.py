"""Flatten ScoreStep results into a tidy, one-row-per-(metric[, control])
table for downstream plotting/aggregation, and merge that table into a
per-(model, dataset) sidecar across repeated runs. Kept out of score.py
since these are pure data-reshaping/merge functions independent of
ScoreStep's own state (config/workspace/IO orchestration).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.schemas.artifacts import (
    ConfidenceInterval,
    ControlComparison,
    EvaluationMetadata,
)

# Keep the seed in the tidy output: bootstrap/control randomness and probe
# initialization are part of the result identity, not just manifest metadata.
_LONG_DF_IDENTITY_FIELDS = tuple(EvaluationMetadata.model_fields)


def build_long_df(
    metadata: EvaluationMetadata | None,
    scores: dict[str, float],
    confidence_intervals: dict[str, ConfidenceInterval] | None,
    control_comparisons: dict[str, dict[str, ControlComparison]] | None,
) -> pl.DataFrame:
    """Flatten a score result into a tidy, one-row-per-(metric[, control])
    table for easy downstream plotting/aggregation.

    Each row carries the full ``EvaluationMetadata`` (model/dataset/task_type/
    split) alongside a ``metric``, so results from several ``ScoreStep`` runs
    (e.g. one per model) can be concatenated with ``pl.concat`` and plotted
    directly (e.g. a bar chart of ``value`` faceted by ``metric`` and
    grouped/colored by ``model_name``), without parsing the run manifest's
    "score"/"bootstrap" sections separately (control comparisons are only
    ever available via ``long_df``/``scores_long.parquet``, in tidy form).

    Rows with ``control`` set to ``None`` are the trained model's own
    point-estimate metrics (``value``, plus ``ci_mean``/``ci_lower``/
    ``ci_upper`` if bootstrap was enabled). Rows with ``control`` set to a
    control name are the paired trained-vs-control bootstrap comparison for
    that (control, metric) pair (``control_mean``, ``diff_mean``, ``p_value``,
    etc.), and are only present when ``ScoreConfig.bootstrap_enabled=True``
    and at least one control was configured.
    """
    # Reuses the same Pydantic serialization every other step uses to
    # persist EvaluationMetadata, rather than a second, hand-maintained
    # field list that silently drifts out of sync whenever EvaluationMetadata
    # gains a field.
    metadata_dict = metadata.model_dump(mode="json") if metadata is not None else {}
    identity = {name: metadata_dict.get(name) for name in _LONG_DF_IDENTITY_FIELDS}

    rows: list[dict[str, Any]] = []
    for metric, value in scores.items():
        ci = (confidence_intervals or {}).get(metric)
        rows.append(
            {
                **identity,
                "metric": metric,
                "control": None,
                "value": value,
                "ci_mean": ci["mean"] if ci else None,
                "ci_lower": ci["lower"] if ci else None,
                "ci_upper": ci["upper"] if ci else None,
                "control_mean": None,
                "diff_mean": None,
                "diff_lower": None,
                "diff_upper": None,
                "p_value": None,
                "p_value_adjusted": None,
            }
        )

    for control_name, metric_comparisons in (control_comparisons or {}).items():
        for metric, comparison in metric_comparisons.items():
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "control": control_name,
                    "value": comparison["trained_mean"],
                    "ci_mean": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "control_mean": comparison["control_mean"],
                    "diff_mean": comparison["diff_mean"],
                    "diff_lower": comparison["diff_lower"],
                    "diff_upper": comparison["diff_upper"],
                    "p_value": comparison["p_value"],
                    "p_value_adjusted": comparison["p_value_adjusted"],
                }
            )

    return pl.DataFrame(rows)


def upsert_long_df(
    path: Path,
    new_df: pl.DataFrame,
    variant: str | None,
    split: str | None = None,
) -> pl.DataFrame:
    """Merge *new_df* into the ``scores_long.parquet`` at *path*, replacing
    any existing rows for the same ``variant`` and ``split`` rather than
    duplicating or discarding
    other variants' rows -- so this file accumulates one comparable table
    across every condition tried for a model+dataset, without the caller
    having to glob/concatenate per-variant files by hand.
    """
    if not path.exists():
        return new_df
    existing = pl.read_parquet(path)
    if "variant" in existing.columns:
        same_variant = (
            pl.col("variant").is_null()
            if variant is None
            else pl.col("variant") == variant
        )
        same_split = (
            pl.col("split").is_null() if split is None else pl.col("split") == split
        )
        if "split" not in existing.columns:
            same_split = pl.lit(split is None or split == "test")
        existing = existing.filter(~(same_variant & same_split))
    return pl.concat([existing, new_df], how="diagonal_relaxed")
