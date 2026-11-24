"""Persisted baseline lookup and scientific identity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.dataset.baseline_data import (
    artifact_identity,
    identity_hash,
    mmseqs_coverage,
)


def artifact_directory(context: dict[str, Any], name: str) -> Path:
    identity = artifact_identity(context, name)
    return (
        Path(context["root"])
        / name
        / f"fit-{identity_hash(identity['fit'])[:12]}"
        / f"test-{identity_hash(identity['test'])[:12]}"
    )


def load_baseline_artifact(
    context: dict[str, Any], name: str
) -> tuple[pl.DataFrame, Path]:
    directory = artifact_directory(context, name)
    manifest_path = directory / "run_manifest.json"
    if context["fit"]["task_type"] == "contact_prediction":
        legacy_manifest_path = directory / "manifest.json"
        if not manifest_path.exists() and legacy_manifest_path.exists():
            manifest_path = legacy_manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared baseline '{name}' at {directory}. "
            "Run run_evaluate.py --baselines with the matching baseline overlay "
            "and dataset/task configuration to generate it."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != artifact_identity(context, name):
        raise ValueError(f"Incompatible baseline manifest: {manifest_path}")
    path = directory / "predictions.parquet"
    artifact_metadata = (
        manifest.get("artifact", manifest)
        if context["fit"]["task_type"] == "contact_prediction"
        else manifest["artifact"]
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact_metadata.get("sha256"):
        raise ValueError(f"Corrupt baseline predictions: {path}")
    frame = pl.read_parquet(path)
    if name == "separation_prior":
        required = {
            "id",
            "record_id",
            "unit_id",
            "split",
            "source",
            "edition",
            "complex_id",
            "target",
            "kind",
            "labels_available",
            "original_i",
            "original_j",
            "labels",
            "scores",
            "length",
            "status",
        }
    else:
        required = {"id", "prediction"}
    if name not in {"amino_acid_frequency", "separation_prior"}:
        required.add("label")
    if context["fit"]["task_type"] in {
        "sequence_classification",
        "token_classification",
    }:
        required.add("probability")
    if name == "mmseqs_knn":
        required.add("fallback")
    if (
        not required.issubset(frame.columns)
        or frame.height != artifact_metadata["rows"]
    ):
        raise ValueError(f"Invalid baseline prediction schema/count: {path}")
    if name == "mmseqs_knn" and artifact_metadata.get("coverage") != mmseqs_coverage(
        frame
    ):
        raise ValueError(f"Invalid MMseqs2 coverage metadata: {manifest_path}")
    return frame, path
