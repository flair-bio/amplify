"""Evaluation config composition, including optional dataset task defaults."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from modules.core.utils.config_loader import load_config, parse_config
from modules.evaluate.src.config.evaluation import BaselineRunConfig, EvaluationConfig
from modules.evaluate.src.config.shared import DatasetWorkspaceConfig
from modules.evaluate.src.proteingym.config import (
    ProteinGymSupervisedConfig,
    ProteinGymZeroShotConfig,
)
from modules.evaluate.src.tasks import available_tasks

RunConfig = (
    EvaluationConfig
    | BaselineRunConfig
    | ProteinGymSupervisedConfig
    | ProteinGymZeroShotConfig
)


def load_run_config(
    paths: Sequence[str | Path],
    overrides: Sequence[str] | None = None,
    *,
    baseline_only: bool = False,
) -> RunConfig:
    """Select the workflow schema before ordinary task-overlay composition."""
    raw = load_config(paths, overrides=overrides)
    reject_removed_options(raw)
    workflow = OmegaConf.select(raw, "workflow", default="evaluation")
    if workflow == "evaluation":
        return (
            load_baseline_config(paths, overrides)
            if baseline_only
            else load_evaluation_config(paths, overrides)
        )
    if baseline_only:
        raise ValueError(
            "--baselines is only supported for ordinary evaluation; configure "
            "ProteinGym reference scorers with baselines instead."
        )
    if workflow == "proteingym_supervised":
        return parse_config(raw, ProteinGymSupervisedConfig)
    if workflow == "proteingym_zero_shot":
        return parse_config(raw, ProteinGymZeroShotConfig)
    raise ValueError(f"Unknown evaluation workflow: {workflow!r}.")


def load_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> EvaluationConfig:
    return parse_config(compose_evaluation_config(path, overrides), EvaluationConfig)


def compose_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load a selected task overlay when no explicit task YAML was supplied.

    Insert the task next to the dataset declaring the default. For the dataset's
    own task, insert before it so dataset-specific protocol settings win. When
    an explicit task override selects a different task, insert after the dataset
    so the selected task's settings replace incompatible dataset defaults.
    Explicit task overlays retain the usual left-to-right merge semantics.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    yaml_config = load_config(paths)
    config = load_config(paths, overrides=overrides)
    reject_removed_options(config)
    default_task = OmegaConf.select(config, "workspace.default_task_type")
    task = OmegaConf.select(config, "workspace.task_type") or default_task
    if OmegaConf.select(yaml_config, "workspace.task_type") is None and task:
        if task not in available_tasks():
            raise ValueError(
                f"Unknown task type '{task}'. Available: {available_tasks()}."
            )
        task_path = (
            Path(__file__).resolve().parents[2] / "configs/tasks" / f"{task}.yaml"
        )
        insertion = len(paths)
        for index, config_path in enumerate(paths):
            declared_task = OmegaConf.select(
                OmegaConf.load(config_path), "workspace.default_task_type"
            )
            if declared_task:
                insertion = index + int(task != declared_task)
        paths.insert(insertion, task_path)
        config = load_config(paths, overrides=overrides)
        reject_removed_options(config)
    return config


def load_baseline_config(
    paths: Sequence[str | Path], overrides: Sequence[str] | None = None
) -> BaselineRunConfig:
    """Compose the dataset/task config with an explicit baseline-only overlay."""
    raw: Any = OmegaConf.to_container(
        compose_evaluation_config(paths, overrides), resolve=True
    )
    # Model and tuning settings do not apply to the baseline route.
    workspace = {
        key: value
        for key, value in raw["workspace"].items()
        if key in DatasetWorkspaceConfig.model_fields
    }
    return BaselineRunConfig.model_validate(
        {
            "workspace": workspace,
            "prepare": raw.get("steps", {}).get("prepare", {}),
            "score": raw.get("steps", {}).get("score", {}),
            "baselines": raw.get("baselines", {}),
            "external_test_sets": raw.get("external_test_sets", []),
            "seed": raw.get("seed", 710019),
            "test_subsets": raw.get("steps", {}).get("predict", {}).get("test_subsets"),
        }
    )


def reject_removed_options(config: DictConfig) -> None:
    if "multi_target" not in config:
        return
    if config.multi_target is not None:
        raise ValueError(
            "multi_target evaluation has been removed. Configure one label_column "
            "and run one scalar-target evaluation per invocation."
        )
    del config["multi_target"]
