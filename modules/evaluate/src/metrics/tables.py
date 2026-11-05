"""Build, merge, and aggregate evaluation score tables."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.contracts.artifacts import (
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
    "score"/"bootstrap" sections separately. Control comparisons are also
    recorded under the manifest's ``bootstrap.control_comparisons``.

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
    """Merge *new_df* into the score table at *path*, replacing
    any existing rows for the same ``variant`` and ``split`` rather than
    duplicating or discarding
    other variants' rows. Pipeline score tables are run-local; callers can
    concatenate them for cross-run comparisons.
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


def aggregate_task_scores(
    model_root: Path,
    run_name: str,
    seed: int,
    dataset_name: str,
    target_columns: tuple[str, ...],
    metric: str,
    *,
    target_key: str = "target",
    count_key: str = "n_targets",
    aggregation_name: str = "unweighted_target_mean",
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Read completed scalar-target runs and compute an equal-weight score mean."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name) or seed < 0:
        raise ValueError("A safe run name and nonnegative seed are required.")
    manifests = {
        prop: model_root / f"{run_name}-{prop}" / f"seed_{seed}" / "run_manifest.json"
        for prop in target_columns
    }
    missing = [prop for prop, path in manifests.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing aggregate-task target runs: {', '.join(missing)}"
        )

    reference_config = reference_metadata = None
    rows = []
    for prop, path in manifests.items():
        manifest = json.loads(path.read_text())
        evaluation = manifest["evaluation"]
        if evaluation["status"] != "complete":
            raise ValueError(f"Aggregate-task target run is incomplete: {prop}")
        config = evaluation["config"]
        metadata = manifest["score"]["metadata"]
        workspace = config["workspace"]
        if (
            workspace["dataset_name"] != dataset_name
            or workspace["task_type"] != "sequence_regression"
            or config["steps"]["prepare"]["label_column"] != prop
            or config["seed"] != seed
            or metadata["seed"] != seed
            or metadata["dataset_name"] != workspace["dataset_name"]
            or metadata["task_type"] != workspace["task_type"]
            or metadata["model_name"] != workspace["model_name"]
            or metadata.get("test_subset") is not None
            or metadata.get("layer_index") is not None
            or f"{metadata['variant']}-{workspace['run_id']}" != f"{run_name}-{prop}"
        ):
            raise ValueError(
                f"Aggregate-task target run has inconsistent identity: {prop}"
            )

        del workspace["run_id"]
        del config["steps"]["prepare"]["label_column"]
        if reference_config is None:
            reference_config, reference_metadata = config, metadata
        elif config != reference_config or metadata != reference_metadata:
            raise ValueError(
                f"Aggregate-task target runs use different protocols: {prop}"
            )
        value = manifest["score"]["scores"][metric]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Aggregate-task target has nonfinite or invalid {metric}: {prop}"
            )
        rows.append(
            {target_key: prop, metric: float(value), "manifest_path": str(path)}
        )

    table = pl.DataFrame(rows)
    summary = {
        "metadata": reference_metadata,
        "run_name": run_name,
        "metric": metric,
        "aggregation": aggregation_name,
        count_key: len(target_columns),
        "value": math.fsum(table[metric].to_list()) / len(target_columns),
    }
    return table, summary
