"""Shared per-assay driver loop for the ProteinGym zero-shot and supervised
entry points: load the reference table, score each assay with a mode-specific
callable, write per-assay + summary artifacts, and aggregate.

Extracted because both entry points otherwise duplicate this loop verbatim,
differing only in how ``model_scores`` is produced for a given assay.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from modules.core.utils.hf_download import download_from_hf, resolve_hf_token
from modules.evaluate.src.proteingym.aggregate import (
    aggregate_proteingym_metrics,
    aggregate_proteingym_scores,
    binarize_labels,
    compute_all_assay_metrics,
    compute_assay_auc,
    compute_assay_metric,
)
from modules.evaluate.src.utils.io import update_run_manifest, write_json, write_parquet

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


def _workspace_scores(
    assays: pl.DataFrame, summary: dict, condition: str
) -> pl.DataFrame:
    """Keep assay metrics distinct from the hierarchy's function/overall means."""
    rows = []
    for assay in assays.iter_rows(named=True):
        metrics = {
            name: assay[name]
            for name in ("spearman", "auc", "mcc", "ndcg", "top_recall")
            if name in assay
        }
        metrics[assay["metric"]] = assay["score"]
        for metric, value in metrics.items():
            rows.append(
                {
                    "scope": "assay",
                    "dms_id": assay["dms_id"],
                    "uniprot_id": assay["uniprot_id"],
                    "coarse_selection_type": assay["coarse_selection_type"],
                    "num_variants": assay["num_variants"],
                    "metric": metric,
                    "control": condition,
                    "value": value,
                }
            )
    primary_metric = summary.get("metric", assays["metric"][0])
    aggregates = {**summary.get("metrics", {}), primary_metric: summary}
    for metric, aggregate in aggregates.items():
        rows.append(
            {
                "scope": "overall",
                "metric": metric,
                "control": condition,
                "value": aggregate["overall"],
            }
        )
        for function, value in aggregate.get("per_category", {}).items():
            rows.append(
                {
                    "scope": "function",
                    "coarse_selection_type": function,
                    "metric": metric,
                    "control": condition,
                    "value": value,
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None)


def _append_condition(frame: pl.DataFrame, path: Path, condition: str) -> Path:
    if path.exists():
        previous = pl.read_parquet(path).filter(pl.col("control") != condition)
        frame = pl.concat([previous, frame], how="diagonal_relaxed")
    return write_parquet(frame, path)


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
    data_dir = _resolve_data_dir(data_dir, data_repo_id, force_download)

    reference = pl.read_parquet(data_dir / "reference.parquet")
    if max_assays is not None:
        reference = reference.head(max_assays)

    assay_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    # Only clinical indels pool variants across genes. Clinical substitutions
    # average per-gene AUCs (one gene per uniprot_id, one Clinical category in
    # the sourced reference), per performance_clinical_benchmarks.py.
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
        if row.get("is_clinical") and row.get("is_indel"):
            pooled_scores.append(model_scores)
            # Normalize before concatenation: clinical genes can use strings or
            # numeric labels, including numeric-only benign pseudocontrols.
            pooled_labels.append(
                binarize_labels(assay[DMS_SCORE_BIN_COLUMN].to_numpy())
            )

    assay_results = pl.DataFrame(assay_rows)
    if run_manifest_path is None:
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
    if run_manifest_path is None:
        write_json(output_dir / "summary.json", summary)
    else:
        root = run_manifest_path.parent
        predictions_path = _append_condition(
            pl.concat(prediction_frames, how="diagonal_relaxed"),
            root / "predictions.parquet",
            condition,
        )
        scores_path = _append_condition(
            _workspace_scores(assay_results, summary, condition),
            root / "scores.parquet",
            condition,
        )
        manifest = json.loads(run_manifest_path.read_text())
        score_section = manifest.get("score", {})
        conditions = score_section.get("condition_scores", {})
        conditions[condition] = {
            "scores": summary,
            "predictions_path": str(predictions_path),
            "scores_path": str(scores_path),
        }
        if condition == "primary":
            score_section["scores"] = summary
        score_section["condition_scores"] = conditions
        score_section["predictions_path"] = str(predictions_path)
        score_section["scores_path"] = str(scores_path)
        update_run_manifest(run_manifest_path, "score", score_section)
    logger.info("Overall (ProteinGym aggregation): %.4f", summary["overall"])
    return summary
