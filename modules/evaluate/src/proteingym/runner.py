"""Shared per-assay driver loop for the ProteinGym zero-shot and supervised
entry points: load the reference table, score each assay with a mode-specific
callable, write per-assay + summary artifacts, and aggregate.

Extracted because both entry points otherwise duplicate this loop verbatim,
differing only in how ``model_scores`` is produced for a given assay.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from modules.core.utils.hf_download import download_from_hf, resolve_hf_token
from modules.evaluate.src.proteingym.aggregate import (
    aggregate_proteingym_metrics,
    aggregate_proteingym_scores,
    compute_all_assay_metrics,
    compute_assay_auc,
    compute_assay_metric,
)
from modules.evaluate.src.utils.io import write_json, write_parquet

logger = logging.getLogger(__name__)

# Repeated multiple times below; centralized so a schema rename can't drift out of sync.
DMS_ID_COLUMN = "dms_id"
DMS_SCORE_COLUMN = "dms_score"
DMS_SCORE_BIN_COLUMN = "dms_score_bin"

# Given a reference row and that assay's DataFrame, return one model score per
# variant, or (scores, metric_override) to report a metric computed some other
# way than compute_assay_metric on those scores (e.g. averaged across CV schemes).
ScoreAssayFn = Callable[
    [dict[str, object], pl.DataFrame], "np.ndarray | tuple[np.ndarray, float]"
]


def _resolve_data_dir(
    data_dir: Path,
    data_repo_id: str | None,
    force_download: bool,
) -> Path:
    """Prefer a local or downloaded HF snapshot, then fall back to ``data_dir``."""
    if data_repo_id is None:
        return data_dir

    if not force_download:
        try:
            return download_from_hf(
                repo_id=data_repo_id,
                local_dir=data_dir,
                repo_type="dataset",
                token=resolve_hf_token(),
                local_files_only=True,
            )
        except Exception as exc:
            logger.info(
                "No local HF snapshot available for %s: %s",
                data_repo_id,
                exc,
            )

    try:
        return download_from_hf(
            repo_id=data_repo_id,
            local_dir=data_dir,
            repo_type="dataset",
            token=resolve_hf_token(),
            force_download=force_download,
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve HF dataset %s; falling back to %s: %s",
            data_repo_id,
            data_dir,
            exc,
        )
        return data_dir


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
    """
    data_dir = _resolve_data_dir(data_dir, data_repo_id, force_download)

    reference = pl.read_parquet(data_dir / "reference.parquet")
    if max_assays is not None:
        reference = reference.head(max_assays)

    assay_rows: list[dict[str, object]] = []
    # Clinical tracks (substitutions and indels): the official benchmark pools
    # every variant across genes into one AUC instead of averaging per-gene
    # AUCs (per-gene sample sizes are too small there, and single-class genes
    # would otherwise yield NaN AUC), per performance_clinical_benchmarks.py's
    # compute_pooled_auc. Stays empty (and unused) for every other track.
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
            result if isinstance(result, tuple) else (result, None)
        )
        assay = assay.with_columns(pl.Series("model_score", model_scores))
        write_parquet(assay, output_dir / f"{dms_id}_scored.parquet")

        has_score = DMS_SCORE_COLUMN in assay.columns
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
            and not row.get("is_clinical")
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
        if row.get("is_clinical"):
            pooled_scores.append(model_scores)
            pooled_labels.append(assay[DMS_SCORE_BIN_COLUMN].to_numpy())

    assay_results = pl.DataFrame(assay_rows)
    write_parquet(assay_results, output_dir / "assay_scores.parquet")

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
        if compute_extra_zero_shot_metrics:
            summary["metrics"] = aggregate_proteingym_metrics(
                assay_results,
                ["spearman", "auc", "mcc", "ndcg", "top_recall"],
                compute_uncertainty=compute_uncertainty,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
    write_json(output_dir / "summary.json", summary)
    logger.info("Overall (ProteinGym aggregation): %.4f", summary["overall"])
    return summary
