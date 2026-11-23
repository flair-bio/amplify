"""Assay-suite result publication using shared atomic artifact primitives."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from modules.evaluate.src.artifacts.files import write_json, write_parquet
from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.proteingym.aggregate import build_workspace_scores


def _append_condition(frame: pl.DataFrame, path: Path, condition: str) -> Path:
    if path.exists():
        previous = pl.read_parquet(path).filter(pl.col("control") != condition)
        frame = pl.concat([previous, frame], how="diagonal_relaxed")
    return write_parquet(frame, path)


def publish_proteingym_results(
    output_dir: Path,
    assays: pl.DataFrame,
    summary: dict,
    predictions: list[pl.DataFrame],
    *,
    run_manifest_path: Path | None,
    condition: str,
) -> None:
    if run_manifest_path is None:
        write_parquet(assays, output_dir / "assay_scores.parquet")
        write_json(output_dir / "summary.json", summary)
        return
    root = run_manifest_path.parent
    predictions_path = _append_condition(
        pl.concat(predictions, how="diagonal_relaxed"),
        root / "predictions.parquet",
        condition,
    )
    scores_path = _append_condition(
        build_workspace_scores(assays, summary, condition),
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
