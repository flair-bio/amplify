"""Cross-step validation/defaulting for EvaluationConfig, kept out of
config.py since it's plain validation logic rather than schema definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.evaluate.src.model.heads import (
    CLASSICAL_HEAD_TYPES,
    TORCH_EMBEDDING_HEAD_TYPES,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.hpo import LOWER_IS_BETTER_METRICS

if TYPE_CHECKING:
    from modules.evaluate.src.config import EvaluationConfig


def validate_cross_step_dependencies(config: EvaluationConfig) -> EvaluationConfig:
    """Validate/apply defaults that depend on more than one step's config,
    mutating *config* in place. Raises ``ValueError`` on an invalid
    combination. Returns *config*, so this doubles as a pydantic
    ``model_validator(mode="after")`` function directly (see config.py)."""
    steps = config.steps
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
    if steps.prepare.packed and task.pairwise:
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
    if task.name == "pseudo_perplexity":
        if steps.mode != "evaluate_as_is":
            raise ValueError(
                "task_type='pseudo_perplexity' requires steps.mode='evaluate_as_is'."
            )
        if "controls" not in steps.predict.model_fields_set:
            steps.predict.controls = []
    if config.workspace.task_type == "categorical_jacobian" and steps.mode != (
        "evaluate_as_is"
    ):
        raise ValueError(
            "task_type='categorical_jacobian' requires steps.mode='evaluate_as_is'."
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
    if "majority_class" in steps.predict.controls and config.workspace.task_type == (
        "sequence_regression"
    ):
        raise ValueError(
            "control='majority_class' is only defined for classification tasks; "
            "use 'train_mean' for sequence_regression."
        )
    if "train_mean" in steps.predict.controls and config.workspace.task_type != (
        "sequence_regression"
    ):
        raise ValueError(
            "control='train_mean' is only defined for sequence_regression tasks; "
            "use 'majority_class' for classification tasks."
        )
    if (
        "separation_prior" in steps.predict.controls
        and config.workspace.task_type
        not in {"contact_prediction", "categorical_jacobian"}
    ):
        raise ValueError(
            "control='separation_prior' is only defined for "
            "task_type in {'contact_prediction', 'categorical_jacobian'}."
        )
    if "metric_aggregation" not in steps.tune.model_fields_set:
        # Otherwise model selection would optimize a different averaging
        # than the one ScoreStep reports.
        steps.tune.metric_aggregation = steps.score.metric_aggregation

    search_config = steps.tune.hyperparameter_search
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
        unsupported_controls = set(steps.predict.controls) & {
            "untrained",
            "random_head",
            "random_both",
        }
        if unsupported_controls:
            raise ValueError(
                f"head_type='{head_type}' does not support control(s) "
                f"{sorted(unsupported_controls)}: use 'random_trunk', "
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
        # Without the cache, TuneStep checks that the loaded native head is a
        # single nn.Linear. Config validation cannot perform that check because
        # model construction happens later in PrepareStep.
    return config
