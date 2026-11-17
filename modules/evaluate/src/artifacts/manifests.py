"""Namespaced evaluation manifest publication."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modules.evaluate.src.artifacts.files import write_json


def update_run_manifest(
    path: Path,
    section: str,
    payload: dict[str, Any],
    *,
    namespace: tuple[str, ...] = (),
) -> Path:
    """Merge ``{section: payload}`` into the run-level manifest at *path*.

    Consolidates what would otherwise be separate JSON sidecars (one per
    step, sprayed across ``preds_path``/``scores_path``/``stats_path``) into
    a single ``run_manifest.json`` at the run root, while preserving
    whichever other steps' sections are already there. *namespace* nests
    step sections, e.g. ``("tests", "casp14")``, without creating another
    manifest for each test subset.
    """
    manifest: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    target = manifest
    for key in namespace:
        nested = target.setdefault(key, {})
        if not isinstance(nested, dict):
            raise ValueError(f"Run manifest namespace {key!r} is not an object")
        target = nested
    target[section] = payload
    return write_json(path, manifest)
