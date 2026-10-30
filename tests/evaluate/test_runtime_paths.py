"""Runtime paths and predictor-owned head: canonical keys, aliases and precedence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from collections.abc import Sequence

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.config.namespaces import canonical_setting
from modules.evaluate.src.config.suites import CONFIG_ROOT
from modules.evaluate.src.dataset.local_data import DataRootError
from modules.evaluate.src.utils.config_loader import compose_evaluation_config
from omegaconf import DictConfig

BASE = CONFIG_ROOT / "config.yaml"


def _composed(
    paths: Sequence[str | Path], overrides: list[str] | None = None
) -> DictConfig:
    """Canonical composition without validating the (partial) base layer."""
    return compose_evaluation_config(paths, overrides)


def _namespace(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "suite": "calibration",
        "model": "amplify_120m",
        "model_collection": None,
        "baselines": False,
        "model_name": None,
        "predictor": None,
        "config_files": [],
        "data_root": None,
        "seed": None,
        "output_root": None,
        "overrides": [],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


# --- runtime.output_root -----------------------------------------------------


def test_output_root_is_canonical_and_maps_to_the_benchmark_root() -> None:
    assert canonical_setting("runtime.output_root=x") == ("runtime", "output_root")
    assert _composed([BASE]).workspace.base_path == "./eval"


def test_legacy_workspace_base_path_is_an_alias_of_output_root() -> None:
    assert canonical_setting("workspace.base_path=/tmp/x") == (
        "runtime",
        "output_root",
    )
    composed = _composed([BASE], overrides=["workspace.base_path=/tmp/legacy"])
    assert composed.workspace.base_path == "/tmp/legacy"


def test_output_root_flag_becomes_runtime_output_root_override() -> None:
    runs = run_evaluate.plan_runs(_namespace(output_root="/tmp/bench"))
    assert "runtime.output_root=/tmp/bench" in runs[0].overrides
    assert not any(o.startswith("workspace.output_dir=") for o in runs[0].overrides)


def test_output_root_flag_conflicts_with_explicit_override() -> None:
    with pytest.raises(SystemExit, match="invalid overrides"):
        run_evaluate.plan_runs(
            _namespace(
                output_root="/tmp/bench",
                overrides=["workspace.base_path=/tmp/other"],
            )
        )


# --- runtime.data_root --------------------------------------------------------


def test_runtime_data_root_defaults_to_null_and_is_recorded() -> None:
    assert _composed([BASE]).data_root is None
    assert _composed([BASE], overrides=["runtime.data_root=/d"]).data_root == "/d"


def test_runtime_data_root_from_yaml_is_applied_to_planned_runs(
    tmp_path: Path,
) -> None:
    layer = tmp_path / "layer.yaml"
    layer.write_text(f"runtime:\n  data_root: {tmp_path / 'empty'}\n")
    (tmp_path / "empty").mkdir()
    with pytest.raises(DataRootError, match="never falls back to the Hub"):
        run_evaluate.plan_runs(_namespace(config_files=[str(layer)]))


def test_data_root_flag_overrides_runtime_data_root_from_yaml(tmp_path: Path) -> None:
    layer = tmp_path / "layer.yaml"
    layer.write_text(f"runtime:\n  data_root: {tmp_path / 'missing'}\n")
    cli_root = tmp_path / "cli"
    (cli_root / "human_calibration").mkdir(parents=True)
    runs = run_evaluate.plan_runs(
        _namespace(config_files=[str(layer)], data_root=str(cli_root))
    )
    mapped = [o for o in runs[0].overrides if o.startswith("dataset.repo_id=")]
    assert mapped == [f"dataset.repo_id={(cli_root / 'human_calibration').resolve()}"]


def test_data_root_flag_conflicts_with_explicit_runtime_override(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="invalid overrides"):
        run_evaluate.plan_runs(
            _namespace(
                data_root=str(tmp_path),
                overrides=[f"runtime.data_root={tmp_path}"],
            )
        )


def test_runtime_data_root_must_be_a_directory(tmp_path: Path) -> None:
    layer = tmp_path / "layer.yaml"
    layer.write_text(f"runtime:\n  data_root: {tmp_path / 'nope'}\n")
    with pytest.raises(SystemExit, match="not a directory"):
        run_evaluate.plan_runs(_namespace(config_files=[str(layer)]))


# --- predictor-owned head -----------------------------------------------------


def test_head_is_canonical_under_predictor() -> None:
    assert canonical_setting("predictor.head.head_type=mlp") == (
        "predictor",
        "head",
        "head_type",
    )


@pytest.mark.parametrize(
    "spelling",
    ["training.head.head_type=torch_linear", "steps.tune.head.head_type=torch_linear"],
)
def test_legacy_head_spellings_alias_to_predictor_head(spelling: str) -> None:
    assert canonical_setting(spelling)[:2] == ("predictor", "head")
    composed = _composed([BASE], overrides=[spelling])
    assert composed.steps.tune.head.head_type == "torch_linear"
    assert "head" not in composed.get("training", {})


def test_head_has_one_spelling_per_layer(tmp_path: Path) -> None:
    layer = tmp_path / "dup.yaml"
    layer.write_text(
        "training:\n  head:\n    head_type: mlp\n"
        "predictor:\n  head:\n    head_type: knn\n"
    )
    with pytest.raises(ValueError, match="predictor.head"):
        _composed([BASE, str(layer)])


def test_predictor_head_wins_over_base_default(tmp_path: Path) -> None:
    layer = tmp_path / "predictor.yaml"
    layer.write_text("predictor:\n  head:\n    head_type: knn\n")
    assert _composed([BASE, str(layer)]).steps.tune.head.head_type == "knn"


def test_unknown_runtime_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="runtime.bogus"):
        canonical_setting("runtime.bogus=1")


@pytest.mark.parametrize(
    "layer",
    [BASE, CONFIG_ROOT / "smoke_test.yaml"],
    ids=["config", "smoke_test"],
)
def test_shipped_base_configs_use_canonical_head_and_runtime_placement(
    layer: Path,
) -> None:
    raw = yaml.safe_load(layer.read_text())
    assert "head" not in raw.get("training", {})
    assert "head" not in raw.get("steps", {}).get("tune", {})
    assert "base_path" not in raw.get("workspace", {})
    assert "output_root" in raw["runtime"]
    if layer == BASE:
        assert "data_root" in raw["runtime"]
