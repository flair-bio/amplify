"""Small file I/O helpers shared across evaluation pipeline steps.

Factored out of ``PredictStep``/``ScoreStep``/``TuneStep`` because each of
those steps independently persists a "reproducibility sidecar" (the
settings -- config + seed, and sometimes other run context -- needed to
reproduce that step's output) alongside its main artifact, using the same
``json.dumps(..., indent=2)`` + ``Path.write_text`` pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write *payload* to *path* as indented JSON, creating parent dirs.

    Returns *path* so callers can assign and use it in one expression.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Write *df* to *path* as parquet, creating parent dirs.

    Mirrors :func:`write_json`'s defensive ``mkdir(parents=True,
    exist_ok=True)`` so callers don't depend solely on
    ``EvaluationWorkspace.setup_directories()`` having already run.
    Returns *path* so callers can assign and use it in one expression.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return path


def _narrowed_dtype(dtype: pl.DataType) -> pl.DataType | None:
    """Return the 32-bit counterpart of a 64-bit numeric dtype, or ``None``
    if *dtype* doesn't need narrowing. Recurses into ``List`` dtypes so
    token_classification's per-token (ragged) predictions/labels are
    narrowed too."""
    if dtype == pl.Float64:
        return pl.Float32
    if dtype == pl.Int64:
        return pl.Int32
    if isinstance(dtype, pl.List):
        inner = _narrowed_dtype(dtype.inner)
        if inner is not None:
            return pl.List(inner)
    return None


def downcast_prediction_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Narrow "prediction"/"label" columns to 32-bit numeric types before
    writing to Parquet, to reduce on-disk file size and I/O throughput.

    Only ``Float64`` -> ``Float32`` (sequence_regression's continuous
    predictions/labels) and ``Int64`` -> ``Int32`` (*_classification's
    discrete class ids) narrowing is applied -- never a float<->int
    conversion, since e.g. truncating a sequence_regression label (a
    continuous target value) to an integer would silently corrupt it. Also
    narrows the inner dtype of ``List`` columns (token_classification's
    per-token predictions/labels). Columns other than "prediction"/"label"
    (e.g. "id", "probability") are left untouched.

    Returns a new DataFrame; does not mutate *df*, so callers can keep using
    the original (full-precision) DataFrame for in-memory metric computation
    while only the copy passed to :func:`write_parquet` is narrowed.
    """
    casts = []
    for name in ("prediction", "label"):
        if name not in df.columns:
            continue
        new_dtype = _narrowed_dtype(df.schema[name])
        if new_dtype is not None:
            casts.append(pl.col(name).cast(new_dtype))

    if not casts:
        return df
    return df.with_columns(casts)


def write_settings_json(
    path: Path, seed: int, config: dict[str, Any] | None, **extra: Any
) -> Path:
    """Write a ``{"seed", "config", ...extra}`` reproducibility sidecar.

    Shared by steps that persist the settings needed to reproduce their
    output alongside their main artifact -- currently only ``TuneStep``'s
    ``tune_config.json``/``probe_head_config.json``, kept beside the model
    checkpoint in ``model_path`` so it stays self-describing if copied out
    on its own. ``PredictStep``/``ScoreStep`` instead use
    :func:`update_run_manifest`, since their settings aren't tied to a
    standalone portable artifact the way the checkpoint is.
    """
    return write_json(path, {"seed": seed, "config": config, **extra})


def update_run_manifest(path: Path, section: str, payload: dict[str, Any]) -> Path:
    """Merge ``{section: payload}`` into the run-level manifest at *path*.

    Consolidates what would otherwise be separate JSON sidecars (one per
    step, sprayed across ``preds_path``/``scores_path``/``stats_path``) into
    a single ``run_manifest.json`` at the run root, while preserving
    whichever other steps' sections are already there -- so Predict/Score
    (which write this file at different times) don't clobber each other.
    """
    manifest: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    manifest[section] = payload
    return write_json(path, manifest)
