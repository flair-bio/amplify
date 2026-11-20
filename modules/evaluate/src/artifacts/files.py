"""Atomic file operations and evaluation serialization."""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep
from typing import Any

import polars as pl


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def cache_lock(
    path: Path, *, timeout: float = 300, poll_interval: float = 0.1
) -> Iterator[None]:
    """Use atomic mkdir, not advisory locks; never reclaim another job's lease."""
    directory = path.with_name(f"{path.name}.d")
    directory.parent.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + timeout
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            if monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {directory}. If its owning job has "
                    "terminated, remove the lock directory before retrying."
                ) from None
            sleep(poll_interval)
    owner = directory / "owner"
    try:
        owner.write_text(f"{socket.gethostname()}:{os.getpid()}\n")
        yield
    finally:
        owner.unlink(missing_ok=True)
        directory.rmdir()


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write *payload* to *path* as indented JSON, creating parent dirs.

    Returns *path* so callers can assign and use it in one expression.
    Atomic: a crash mid-write leaves the previous file intact.
    """
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(dict(payload), indent=2))
    return path


def write_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Write *df* to *path* as parquet, creating parent dirs.

    Mirrors :func:`write_json`'s defensive ``mkdir(parents=True,
    exist_ok=True)`` so callers don't depend solely on
    ``EvaluationWorkspace.setup_directories()`` having already run.
    Returns *path* so callers can assign and use it in one expression.
    """
    with atomic_output(path) as temporary:
        df.write_parquet(temporary)
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

    Keep fitting settings beside portable checkpoints. Prediction and scoring
    metadata instead belong in the shared run manifest.
    """
    return write_json(path, {"seed": seed, "config": config, **extra})
