"""Tests for modules.evaluate.src.utils.io.update_run_manifest."""

from __future__ import annotations

import json

from modules.evaluate.src.utils.io import update_run_manifest


def test_update_run_manifest_creates_preserves_and_overwrites_sections(tmp_path):
    path = tmp_path / "run_manifest.json"

    result_path = update_run_manifest(path, "predict", {"seed": 1})

    assert result_path == path
    assert json.loads(path.read_text()) == {"predict": {"seed": 1}}

    update_run_manifest(path, "score", {"scores": {"accuracy": 0.9}})
    manifest = json.loads(path.read_text())
    assert manifest["predict"] == {"seed": 1}
    assert manifest["score"] == {"scores": {"accuracy": 0.9}}

    update_run_manifest(path, "predict", {"seed": 2})

    manifest = json.loads(path.read_text())
    assert manifest["score"] == {"scores": {"accuracy": 0.9}}
    assert manifest["predict"] == {"seed": 2}
