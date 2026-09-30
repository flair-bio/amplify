"""First iteration of tests for the data pipeline config system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig

from modules.data.src.config import DataPipelineConfig
from modules.core.utils.config_loader import load_and_parse, load_config

CONFIGS_DIR = Path(__file__).parents[2] / "modules" / "data" / "configs"
MAIN_CONFIG = CONFIGS_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_load_config_returns_dictconfig():
    """load_config returns an OmegaConf DictConfig for a valid YAML file."""
    cfg = load_config(MAIN_CONFIG)
    assert isinstance(cfg, DictConfig)
    assert "dataset" in cfg
    assert "steps" in cfg


# ---------------------------------------------------------------------------
# Single YAML parsing
# ---------------------------------------------------------------------------


def test_single_yaml_parsed_correctly():
    """Parsing config.yaml produces a fully validated DataPipelineConfig."""
    config = load_and_parse(MAIN_CONFIG, DataPipelineConfig)

    assert isinstance(config, DataPipelineConfig)
    assert config.dataset.base_path == "./datasets"
    assert config.steps.download.enabled is False
    assert config.steps.preprocess.enabled is False
    assert config.steps.score.enabled is False


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides_applied_correctly():
    """Dotlist overrides passed via the CLI take precedence over YAML values."""
    config = load_and_parse(
        MAIN_CONFIG,
        DataPipelineConfig,
        overrides=[
            "steps.download.enabled=false",
            "dataset.base_path=/tmp/override_data",
        ],
    )

    assert config.steps.download.enabled is False
    assert config.dataset.base_path == "/tmp/override_data"


# ---------------------------------------------------------------------------
# YAML file merging
# ---------------------------------------------------------------------------


def test_merge_yaml_files(tmp_path: Path):
    """Later YAML files override keys from earlier ones; unique keys are preserved."""
    base_yaml = tmp_path / "base.yaml"
    override_yaml = tmp_path / "override.yaml"

    base_yaml.write_text(
        "dataset:\n"
        "  name: bfd\n"
        "  base_path: /base/data\n"
        "steps:\n"
        "  download:\n"
        "    enabled: false\n"
        "  preprocess:\n"
        "    enabled: true\n"
    )
    override_yaml.write_text(
        "dataset:\n"
        "  name: bfd\n"
        "  base_path: /override/data\n"
        "steps:\n"
        "  download:\n"
        "    enabled: false\n"
        "  preprocess:\n"
        "    enabled: false\n"
    )

    config = load_and_parse([base_yaml, override_yaml], DataPipelineConfig)

    # override_yaml wins for these keys
    assert config.steps.download.enabled is False
    assert config.steps.preprocess.enabled is False
    # base_yaml value is preserved for keys not in override_yaml
    assert config.dataset.base_path == "/override/data"


# ---------------------------------------------------------------------------
# run_data.py toy pipeline
# ---------------------------------------------------------------------------


def test_run_data_main_loads_and_builds_dataset(monkeypatch):
    """run_data.main() loads config, validates it, and builds a dataset instance."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_data",
            str(MAIN_CONFIG),
            "steps.download.enabled=false",
            "steps.preprocess.enabled=false",
            "steps.cluster.enabled=false",
            "steps.score.enabled=false",
            "steps.assemble.enabled=false",
            "steps.stats.enabled=false",
        ],
    )

    from modules.data.src.run_data import main

    # Should complete without raising — no I/O is triggered, only object init.
    main()
