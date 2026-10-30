"""Config namespaces: canonical documents, legacy aliases, and deterministic precedence."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.config.namespaces import (
    canonical_setting,
    reject_conflicting_settings,
)
from modules.evaluate.src.config.suites import CONFIG_ROOT
from modules.evaluate.src.utils.config_loader import (
    compose_evaluation_config,
    load_evaluation_config,
)

BASE = CONFIG_ROOT / "config.yaml"

LEGACY_LAYER = """\
workspace:
  dataset_name: toy
  dataset_repo_id: org/toy
  target_type: categorical
  label_format: scalar
  num_labels: 2
  task_type: sequence_classification
steps:
  score:
    n_samples: 5
"""

CANONICAL_LAYER = """\
dataset:
  name: toy
  repo_id: org/toy
  target_type: categorical
  label_format: scalar
  num_labels: 2
task:
  type: sequence_classification
scoring:
  n_samples: 5
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _composed(paths: list[Path], overrides: list[str] | None = None) -> dict:
    return OmegaConf.to_container(
        compose_evaluation_config(paths, overrides), resolve=True
    )


def test_canonical_and_legacy_documents_compose_identically(tmp_path: Path) -> None:
    legacy = _composed([BASE, _write(tmp_path, "legacy.yaml", LEGACY_LAYER)])
    canonical = _composed([BASE, _write(tmp_path, "canon.yaml", CANONICAL_LAYER)])
    assert legacy == canonical
    assert (
        EvaluationConfig.model_validate(legacy).model_dump()
        == EvaluationConfig.model_validate(canonical).model_dump()
    )


def test_legacy_keys_emit_a_migration_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _composed([BASE, _write(tmp_path, "legacy.yaml", LEGACY_LAYER)])
    text = caplog.text
    assert "deprecated legacy config key" in text
    assert "workspace.dataset_name -> dataset.name" in text
    assert "workspace.dataset_repo_id" in text and "dataset.repo_id" in text


def test_canonical_documents_emit_no_migration_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _composed([BASE, _write(tmp_path, "canon.yaml", CANONICAL_LAYER)])
    assert "deprecated" not in caplog.text


def test_same_setting_under_both_spellings_in_one_layer_is_rejected(
    tmp_path: Path,
) -> None:
    both = _write(
        tmp_path,
        "both.yaml",
        "steps:\n  score:\n    n_samples: 5\nscoring:\n  n_samples: 6\n",
    )
    with pytest.raises(ValueError, match="scoring.n_samples.*both"):
        compose_evaluation_config([BASE, both])


def test_later_layers_win_regardless_of_spelling(tmp_path: Path) -> None:
    legacy_base = _write(tmp_path, "a.yaml", "steps:\n  score:\n    n_samples: 1\n")
    canonical_over = _write(tmp_path, "b.yaml", "scoring:\n  n_samples: 2\n")
    assert (
        _composed([BASE, legacy_base, canonical_over])["steps"]["score"]["n_samples"]
        == 2
    )
    assert (
        _composed([BASE, canonical_over, legacy_base])["steps"]["score"]["n_samples"]
        == 1
    )


def test_dotlist_overrides_accept_both_spellings(tmp_path: Path) -> None:
    raw = _composed(
        [BASE, _write(tmp_path, "canon.yaml", CANONICAL_LAYER)],
        ["model.revision=" + "a" * 40, "steps.score.n_samples=7"],
    )
    assert raw["workspace"]["model_revision"] == "a" * 40
    assert raw["steps"]["score"]["n_samples"] == 7


@pytest.mark.parametrize(
    "layer",
    [
        "bogus: 1\n",
        "scoring:\n  n_sampels: 3\n",
        "steps:\n  nope: 1\n",
        "model:\n  revisoin: abc\n",
    ],
)
def test_unknown_namespaces_and_fields_are_rejected(tmp_path: Path, layer: str) -> None:
    path = _write(tmp_path, "bad.yaml", layer)
    with pytest.raises((ValidationError, ValueError)):
        load_evaluation_config([BASE, path])


def test_canonical_setting_maps_legacy_paths() -> None:
    assert canonical_setting("steps.score.n_samples=3") == ("scoring", "n_samples")
    assert canonical_setting("scoring.n_samples=3") == ("scoring", "n_samples")
    assert canonical_setting("seed=1") == ("runtime", "seed")
    assert canonical_setting("workspace.model_name=x") == ("model", "name")


def test_conflicting_cli_settings_are_rejected_not_order_dependent() -> None:
    with pytest.raises(ValueError, match="runtime.seed"):
        reject_conflicting_settings(["seed=1", "runtime.seed=2"])
    reject_conflicting_settings(["seed=1", "scoring.n_samples=2"])


def test_namespace_documents_keep_legacy_workspace_run_fields(tmp_path: Path) -> None:
    raw = _composed([BASE], ["workspace.run_id=abc", "workspace.base_path=/tmp/x"])
    assert raw["workspace"]["run_id"] == "abc"
    assert raw["workspace"]["base_path"] == "/tmp/x"
