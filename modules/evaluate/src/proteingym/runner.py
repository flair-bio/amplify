"""Shared per-assay driver loop for the ProteinGym zero-shot and supervised
entry points: load the reference table, score each assay with a mode-specific
callable, write per-assay + summary artifacts, and aggregate.

Extracted because both entry points otherwise duplicate this loop verbatim,
differing only in how ``model_scores`` is produced for a given assay.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from modules.evaluate.src.artifacts.files import write_parquet
from modules.evaluate.src.artifacts.proteingym import publish_proteingym_results
from modules.evaluate.src.proteingym.aggregate import (
    aggregate_proteingym_metrics,
    aggregate_proteingym_scores,
    binarize_labels,
    compute_all_assay_metrics,
    compute_assay_auc,
    compute_assay_metric,
)
from modules.evaluate.src.proteingym.data import resolve_data_dir

logger = logging.getLogger(__name__)
DMS_ID_COLUMN = "dms_id"
DMS_SCORE_COLUMN = "dms_score"
DMS_SCORE_BIN_COLUMN = "dms_score_bin"


@dataclass(frozen=True)
class AssayResult:
    """Assay predictions with an optional protocol-specific score override."""

    predictions: np.ndarray
    reported_score: float | None = None


ScoreAssayFn = Callable[[dict[str, object], pl.DataFrame], "np.ndarray | AssayResult"]


def run_assay_loop(
    data_dir: Path,
    output_dir: Path,
    max_assays: int | None,
    score_assay: ScoreAssayFn,
    data_repo_id: str | None = None,
    force_download: bool = False,
    compute_uncertainty: bool = False,
    n_bootstrap: int = 10000,
    bootstrap_seed: int = 0,
    compute_extra_zero_shot_metrics: bool = False,
    run_manifest_path: Path | None = None,
    condition: str = "primary",
) -> dict[str, object]:
    """Score every assay in ``data_dir``'s reference table and aggregate the results.

    If ``data_repo_id`` is set, use a local HF snapshot when available, then
    try downloading the repository (the layout ``source_proteingym.py``
    uploads: ``reference.parquet`` plus ``data/<dms_id>.parquet``). If HF is
    unavailable, read the configured ``data_dir`` as a fallback.

    ``compute_extra_zero_shot_metrics`` additionally computes and aggregates
    MCC/NDCG/Top_recall (alongside the primary Spearman/AUC in ``summary``),
    matching the official DMS zero-shot benchmark's full metric set. Clinical
    assays are excluded (the official clinical benchmark reports AUC only)
    and the supervised entry point leaves this off (its official metrics are
    Spearman/MSE only).

    Workspace runs consolidate all assays/conditions into predictions.parquet
    and scores.parquet beside ``run_manifest_path``. Without it, retain the
    legacy per-assay, summary.json and assay_scores.parquet output contract.
    """
    data_dir = resolve_data_dir(data_dir, data_repo_id, force_download)
    reference = pl.read_parquet(data_dir / "reference.parquet")
    if max_assays is not None:
        reference = reference.head(max_assays)
    assay_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    for row in reference.iter_rows(named=True):
        dms_id = row[DMS_ID_COLUMN]
        assay_path = data_dir / "data" / f"{dms_id}.parquet"
        if not assay_path.exists():
            logger.warning("Skipping %s: %s not found", dms_id, assay_path)
            continue
        assay = pl.read_parquet(assay_path)
        result = score_assay(row, assay)
        model_scores, metric_override = (
            (result.predictions, result.reported_score)
            if isinstance(result, AssayResult)
            else (result, None)
        )
        assay = assay.with_columns(pl.Series("model_score", model_scores))
        has_score = DMS_SCORE_COLUMN in assay.columns
        if run_manifest_path is None:
            write_parquet(assay, output_dir / f"{dms_id}_scored.parquet")
        else:
            labels = (
                assay[DMS_SCORE_COLUMN].to_numpy()
                if has_score
                else binarize_labels(assay[DMS_SCORE_BIN_COLUMN].to_numpy())
            )
            if not has_score and row.get("is_clinical"):
                labels = 1 - labels
            predictions = assay.rename({"model_score": "prediction"}).with_columns(
                pl.lit(str(dms_id)).alias("dms_id"),
                pl.lit(condition).alias("control"),
                pl.Series("label", labels, dtype=pl.Float64),
            )
            if "id" not in predictions.columns:
                predictions = predictions.with_columns(
                    pl.Series("id", [f"{dms_id}:{i}" for i in range(assay.height)])
                )
            prediction_frames.append(predictions)
        if metric_override is None:
            metric_name, metric_value = compute_assay_metric(
                model_scores,
                dms_scores=assay[DMS_SCORE_COLUMN].to_numpy() if has_score else None,
                dms_score_bin=assay[DMS_SCORE_BIN_COLUMN].to_numpy()
                if not has_score
                else None,
                flip_binary_labels=bool(row.get("is_clinical")),
            )
        else:
            metric_name, metric_value = (
                "spearman" if has_score else "auc",
                metric_override,
            )
        assay_row: dict[str, object] = {
            DMS_ID_COLUMN: dms_id,
            "uniprot_id": row["uniprot_id"],
            "coarse_selection_type": row["coarse_selection_type"],
            "metric": metric_name,
            "score": metric_value,
            "num_variants": assay.height,
        }
        if (
            compute_extra_zero_shot_metrics
            and metric_override is None
            and (not row.get("is_clinical"))
        ):
            assay_row.update(
                compute_all_assay_metrics(
                    model_scores,
                    dms_scores=assay[DMS_SCORE_COLUMN].to_numpy()
                    if has_score
                    else None,
                    dms_score_bin=assay[DMS_SCORE_BIN_COLUMN].to_numpy()
                    if DMS_SCORE_BIN_COLUMN in assay.columns
                    else None,
                )
            )
        assay_rows.append(assay_row)
        logger.info(
            "Scored %s: %s=%.4f (n=%d)", dms_id, metric_name, metric_value, assay.height
        )
        if row.get("is_clinical") and row.get("is_indel"):
            pooled_scores.append(model_scores)
            pooled_labels.append(
                binarize_labels(assay[DMS_SCORE_BIN_COLUMN].to_numpy())
            )
    assay_results = pl.DataFrame(assay_rows)
    if pooled_scores:
        overall = compute_assay_auc(
            np.concatenate(pooled_scores),
            np.concatenate(pooled_labels),
            flip_labels=True,
        )
        summary = {"overall": overall, "metric": "auc", "aggregation": "pooled"}
    else:
        summary = aggregate_proteingym_scores(
            assay_results,
            compute_uncertainty=compute_uncertainty,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        if "is_clinical" in reference.columns and reference["is_clinical"].all():
            summary.update(
                metric="auc",
                aggregation="per_gene",
                num_genes=assay_results.filter(pl.col("score").is_finite()).height,
            )
        elif compute_extra_zero_shot_metrics:
            summary["metrics"] = aggregate_proteingym_metrics(
                assay_results,
                ["spearman", "auc", "mcc", "ndcg", "top_recall"],
                compute_uncertainty=compute_uncertainty,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
    publish_proteingym_results(
        output_dir,
        assay_results,
        summary,
        prediction_frames,
        run_manifest_path=run_manifest_path,
        condition=condition,
    )
    logger.info("Overall (ProteinGym aggregation): %.4f", summary["overall"])
    return summary
