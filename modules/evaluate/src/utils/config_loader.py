"""Evaluation config composition, including optional dataset task defaults."""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

from modules.core.utils.config_loader import load_config, parse_config
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.tasks import available_tasks


def load_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> EvaluationConfig:
    """Load a selected task overlay when no explicit task YAML was supplied.

    Insert the task immediately before the dataset declaring the default, so
    base settings are overridden by task behavior, and dataset settings and
    later overlays can still override that behavior. Explicit task overlays
    retain the usual left-to-right merge semantics.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    yaml_config = load_config(paths)
    config = load_config(paths, overrides=overrides)
    default_task = OmegaConf.select(config, "workspace.default_task_type")
    task = OmegaConf.select(config, "workspace.task_type") or default_task
    if OmegaConf.select(yaml_config, "workspace.task_type") is None and task:
        if task not in available_tasks():
            raise ValueError(
                f"Unknown task type '{task}'. "
                f"Available: {available_tasks()}."
            )
        task_path = (
            Path(__file__).resolve().parents[2]
            / "configs/tasks"
            / f"{task}.yaml"
        )
        insertion = len(paths)
        for index, config_path in enumerate(paths):
            if OmegaConf.select(
                OmegaConf.load(config_path), "workspace.default_task_type"
            ):
                insertion = index
        paths.insert(insertion, task_path)
        config = load_config(paths, overrides=overrides)
    return parse_config(config, EvaluationConfig)
