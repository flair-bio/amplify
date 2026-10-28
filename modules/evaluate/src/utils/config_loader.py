"""Evaluation config composition.

Task behavior is always selected explicitly (by a suite or a task overlay).
Dataset YAML files describe data only and never choose a task.
"""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from modules.core.utils.config_loader import load_config, parse_config
from modules.evaluate.src.config import EvaluationConfig


def load_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> EvaluationConfig:
    return parse_config(compose_evaluation_config(path, overrides), EvaluationConfig)


def compose_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Compose YAML files left-to-right with optional dotlist overrides."""
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    return load_config(paths, overrides=overrides)
