"""Evaluation configuration composition, validation, and source helpers."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from omegaconf import DictConfig, OmegaConf

from modules.core.utils.config_loader import load_config, parse_config

if TYPE_CHECKING:
    from modules.evaluate.src.config import (
        BaselineRunConfig,
        EvaluationConfig,
        TestSubsets,
    )
    from modules.evaluate.src.proteingym.config import (
        ProteinGymSupervisedConfig,
        ProteinGymZeroShotConfig,
    )

RunConfig = Union[
    "EvaluationConfig",
    "BaselineRunConfig",
    "ProteinGymSupervisedConfig",
    "ProteinGymZeroShotConfig",
]


def validate_directory_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "Output labels must start with a letter or digit and contain "
            "only letters, digits, '.', '_', or '-'."
        )
    return value


def validate_metric_group_column(value: str | None) -> str | None:
    if value is not None and (
        not value.strip() or value in {"id", "prediction", "label", "probability"}
    ):
        raise ValueError("group_column must name a non-reserved source column.")
    return value


def validate_test_subsets(value: TestSubsets | None) -> TestSubsets:
    if value is None:
        return {}
    for name, filters in value.items():
        validate_directory_label(name)
        if not filters or any(not column.strip() for column in filters):
            raise ValueError("Test subsets require nonempty column-equality filters.")
        if any(
            isinstance(item, float) and not math.isfinite(item)
            for item in filters.values()
        ):
            raise ValueError("Test subset filter values must be finite.")
    return value


def dataset_source_identity(
    dataset_id: str, config_name: str | None = None, revision: str | None = None
) -> str:
    """Distinguish configurations/revisions sharing the same repo or suite root."""
    if config_name is None and revision is None:
        return dataset_id
    return json.dumps([dataset_id, config_name, revision])


def validate_cross_step_dependencies(config: EvaluationConfig) -> EvaluationConfig:
    """Validate/apply defaults that depend on more than one step's config,
    mutating *config* in place. Raises ``ValueError`` on an invalid
    combination. Returns *config*, so this doubles as a pydantic
    ``model_validator(mode="after")`` function directly (see config.py)."""
    from modules.evaluate.src.config import (
        CLASSICAL_HEAD_TYPES,
        TORCH_EMBEDDING_HEAD_TYPES,
    )
    from modules.evaluate.src.metrics.metrics import LOWER_IS_BETTER_METRICS
    from modules.evaluate.src.tasks import get_task

    steps = config.steps
    task = get_task(config.workspace.task_type)
    steps.score.validate_task(config.workspace.task_type)
    if (
        steps.tune.hyperparameter_search.group_column is not None
        and config.workspace.task_type != "sequence_regression"
    ):
        raise ValueError(
            "tune.hyperparameter_search.group_column is supported for "
            "sequence_regression only."
        )
    if task.fixed_protocol:
        task.validate_config(config)
        return config
    steps.tune.head.validate_task(config.workspace.task_type)
    if steps.layerwise.enabled:
        # The dedicated trainer compares frozen representations with linear
        # heads; generic HPO and seed-variance paths do not manage head sets.
        if steps.mode != "tune_probe" or steps.tune.head.head_type != "torch_linear":
            raise ValueError(
                "layerwise probing requires tune_probe and head_type='torch_linear'."
            )
        if steps.tune.hyperparameter_search.enabled:
            raise ValueError(
                "layerwise probing does not support hyperparameter search."
            )
        if steps.tune.variance_seeds:
            raise ValueError("layerwise probing does not support variance_seeds.")
        if not task.supports_layerwise:
            raise ValueError(
                "layerwise probing supports sequence and token tasks only."
            )
    if steps.mode == "evaluate_as_is" and steps.prepare.embedding_cache.enabled:
        raise ValueError(
            "steps.mode='evaluate_as_is' requires "
            "steps.prepare.embedding_cache.enabled=False."
        )
    if steps.mode == "finetune":
        if steps.tune.head.head_type != "mlp":
            raise ValueError("steps.mode='finetune' requires head_type='mlp'.")
        if steps.prepare.embedding_cache.enabled:
            raise ValueError(
                "steps.mode='finetune' requires "
                "steps.prepare.embedding_cache.enabled=False."
            )
    task = get_task(config.workspace.task_type)
    # Calibration's fixed-width per-row labels cannot be packed per token.
    if steps.prepare.packed and not task.supports_packing:
        raise ValueError(f"Packed evaluation does not support task_type='{task.name}'.")
    if steps.prepare.max_tokens_per_batch is not None and not steps.prepare.packed:
        raise ValueError("max_tokens_per_batch requires steps.prepare.packed=True.")
    task.validate_dataset_config(
        config.workspace.target_type,
        config.workspace.label_format,
        config.workspace.dataset_name,
    )
    task.resolve_num_labels(
        config.workspace.num_labels,
        config.workspace.dataset_name,
    )
    if steps.mode not in task.evaluation_modes:
        raise ValueError(
            f"task_type='{task.name}' requires steps.mode='evaluate_as_is'."
        )
    metrics = steps.score.metrics
    if metrics is None:
        metrics = task.default_metrics()
    if task.metrics_requiring_probabilities & set(metrics) and not (
        steps.predict.collect_probabilities
    ):
        # Keep metric defaults "all supported" without extra user config.
        steps.predict.collect_probabilities = True
    if steps.predict.collect_probabilities and task.is_ragged:
        raise ValueError(
            f"steps.predict.collect_probabilities=True is not supported for "
            f"task_type='{config.workspace.task_type}': its per-example "
            "predictions are variable-length, so a (sequence, class) "
            "probability matrix cannot be row-aligned to them. No ragged task "
            "declares a probability-based metric."
        )
    if "scrambled_sequences" in steps.predict.controls and task.is_ragged:
        raise ValueError(
            "control='scrambled_sequences' is not supported for ragged task "
            f"type='{config.workspace.task_type}': sequence tokens are shuffled "
            "without jointly permuting their position-aligned labels."
        )
    unsupported_baselines = (
        set[str](steps.predict.baselines) - task.prediction_baselines
    )
    if unsupported_baselines:
        raise ValueError(
            f"Baselines {sorted(unsupported_baselines)} are not defined for task_type='{task.name}'."
        )
    unsupported_controls = set[str](steps.predict.controls) - task.supported_controls
    if unsupported_controls:
        raise ValueError(
            f"Controls {sorted(unsupported_controls)} are not supported for task_type='{task.name}'."
        )
    if "metric_aggregation" not in steps.tune.model_fields_set:
        # Otherwise model selection would optimize a different averaging
        # than the one scoring reports.
        steps.tune.metric_aggregation = steps.score.metric_aggregation

    search_config = steps.tune.hyperparameter_search
    if (
        search_config.enabled
        and (
            search_config.search_space.hidden_size is not None
            or search_config.search_space.hidden_depth is not None
            or search_config.search_space.batch_size is not None
        )
        and (
            steps.mode != "tune_probe"
            or steps.tune.head.head_type != "mlp"
            or not steps.prepare.embedding_cache.enabled
            or steps.layerwise.enabled
        )
    ):
        dimension = (
            "hidden_size"
            if search_config.search_space.hidden_size is not None
            else (
                "hidden_depth"
                if search_config.search_space.hidden_depth is not None
                else "batch_size"
            )
        )
        raise ValueError(
            f"{dimension} search requires a cached MLP probe in tune_probe mode."
        )
    if steps.tune.refit_on_train_validation and (
        steps.mode != "tune_probe"
        or not steps.prepare.embedding_cache.enabled
        or steps.layerwise.enabled
    ):
        raise ValueError(
            "refit_on_train_validation requires an ordinary cached probe "
            "in tune_probe mode; live/native and layerwise refit are not supported."
        )
    if (
        steps.prepare.split.refit_additional_splits
        and not steps.tune.refit_on_train_validation
    ):
        raise ValueError(
            "steps.prepare.split.refit_additional_splits requires "
            "steps.tune.refit_on_train_validation=true."
        )
    if search_config.enabled and search_config.direction == "maximize":
        # An explicit 'maximize' for a lower-is-better metric would select the
        # WORST trial as "best" with no useful result.
        resolved_metric = search_config.metric or task.default_metrics()[0]
        if resolved_metric in LOWER_IS_BETTER_METRICS:
            raise ValueError(
                f"steps.tune.hyperparameter_search.direction='maximize' contradicts "
                f"metric='{resolved_metric}', which is lower-is-better "
                f"({sorted(LOWER_IS_BETTER_METRICS)}). Omit direction to auto-select "
                "'minimize', or set direction='minimize' explicitly."
            )

    if steps.prepare.embedding_cache.enabled and steps.mode != "tune_probe":
        # A trained trunk's output changes every epoch, so caching it
        # would freeze stale features into every subsequent trial.
        raise ValueError(
            "steps.prepare.embedding_cache.enabled=True requires "
            "steps.mode='tune_probe' (a frozen trunk); the trunk's "
            "output must be fixed for caching it to be valid."
        )
    head_type = steps.tune.head.head_type
    if head_type in CLASSICAL_HEAD_TYPES:
        if task.is_ragged:
            raise ValueError(
                f"head_type='{head_type}' is not supported for ragged task "
                f"type='{config.workspace.task_type}': sklearn-compatible "
                "probes require one fixed-size feature vector and label per "
                "example, while token-level tasks have variable-length rows."
            )
        if steps.mode != "tune_probe":
            raise ValueError(
                f"head_type='{head_type}' has no gradient, so "
                "steps.mode must be 'tune_probe'."
            )
        if not steps.prepare.embedding_cache.enabled:
            raise ValueError(
                f"head_type='{head_type}' requires "
                "steps.prepare.embedding_cache.enabled=True: it fits on "
                "fixed-size pooled embeddings, not token batches."
            )
        # These controls need a randomly re-initializable head, not
        # available for a fit-once classical estimator.
        unsupported_head_controls = set(steps.predict.controls) & {
            "untrained",
            "random_head",
            "random_both",
        }
        if unsupported_head_controls:
            raise ValueError(
                f"head_type='{head_type}' does not support control(s) "
                f"{sorted(unsupported_head_controls)}: use 'random_trunk', "
                "'scrambled_sequences', and/or 'scrambled_labels' instead."
            )

    if head_type in TORCH_EMBEDDING_HEAD_TYPES:
        if steps.mode != "tune_probe":
            raise ValueError(
                "head_type='torch_linear' requires steps.mode='tune_probe' "
                "because it probes a frozen trunk."
            )
        # token_classification is exempt from requiring embedding_cache:
        # every task-model pairing this repo ships
        # (AMPLIFYForTokenClassification, transformers'
        # EsmForTokenClassification) already builds a plain nn.Linear head
        # with no cache needed. sequence_classification/regression heads
        # aren't guaranteed pure-linear (e.g. ESM's EsmClassificationHead
        # has a hidden dense+tanh projection), so those still require the
        # embedding cache to construct the standalone linear probe. Setting
        # embedding_cache.enabled=True for token_classification anyway is
        # also valid: it trains an equivalent standalone linear probe over
        # cached per-token embeddings instead of the native head.
    return config


def load_run_config(
    paths: Sequence[str | Path],
    overrides: Sequence[str] | None = None,
    *,
    baseline_only: bool = False,
) -> RunConfig:
    """Select the workflow schema before ordinary task-overlay composition."""
    from modules.evaluate.src.proteingym.config import (
        ProteinGymSupervisedConfig,
        ProteinGymZeroShotConfig,
    )

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
    from modules.evaluate.src.config import EvaluationConfig

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
    from modules.evaluate.src.tasks import available_tasks

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
    from modules.evaluate.src.config import BaselineRunConfig, DatasetWorkspaceConfig

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
