"""Engine factory: resolves a validated config to the generic-runner task that runs it.

This module is the single place that knows which task owners run on the generic
lifecycle. The CLI asks :func:`resolve_engine` and only falls back to the legacy
pipeline when it returns ``None``. Adding a family means adding an
:class:`EngineSpec` here, not a branch in the CLI or the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.evaluation.contracts import TaskDefinition
from modules.evaluate.src.predictors.contact import (
    ContactAttentionPredictor,
    ContactEmbeddingPredictor,
    ContactJacobianPredictor,
    ContactPredictor,
)
from modules.evaluate.src.tasks.contact import ContactTask
from modules.evaluate.src.tasks.frozen_probe import FrozenProbeTask
from modules.evaluate.src.tasks.masked_lm import MASKED_LM_TASK_TYPES, MaskedLMTask

SUPPORTED_TASK_TYPES = frozenset(
    {
        "sequence_classification",
        "sequence_multilabel_classification",
        "sequence_regression",
        "token_classification",
    }
)


@dataclass(frozen=True)
class EngineSpec:
    """One generic-runner task: which configs it covers and how it is built."""

    name: str
    supports: Callable[[EvaluationConfig], bool]
    build: Callable[[EvaluationConfig], TaskDefinition]


def _supports_frozen_probe(config: EvaluationConfig) -> bool:
    """Supervised sequence and token tasks in any of the three supervised modes.

    Layerwise, live (uncached), packed and the probe variants are selected by the
    task's configuration; the engine claims the job by task type and mode alone.
    """
    return config.workspace.task_type in SUPPORTED_TASK_TYPES and config.steps.mode in {
        "tune_probe",
        "finetune",
        "evaluate_as_is",
    }


def _supports_masked_lm(config: EvaluationConfig) -> bool:
    """Zero-shot calibration and pseudo-perplexity on the pretrained MLM, flat only."""
    return (
        config.workspace.task_type in MASKED_LM_TASK_TYPES
        and config.steps.mode == "evaluate_as_is"
        and not config.steps.layerwise.enabled
    )


def _contact_engine(
    predictor_type: type[ContactPredictor],
) -> EngineSpec:
    """One contact head: the registry binds the predictor, the task stays generic."""

    def supports(config: EvaluationConfig) -> bool:
        return (
            config.workspace.task_type == "contact_prediction"
            and config.steps.tune.head.head_type == predictor_type.head_type
        )

    return EngineSpec(
        name=f"{ContactTask.name}/{predictor_type.head_type}",
        supports=supports,
        build=lambda config: ContactTask(config, predictor_type),
    )


ENGINES: tuple[EngineSpec, ...] = (
    EngineSpec(
        name=FrozenProbeTask.name,
        supports=_supports_frozen_probe,
        build=FrozenProbeTask,
    ),
    EngineSpec(
        name=MaskedLMTask.name,
        supports=_supports_masked_lm,
        build=MaskedLMTask,
    ),
    _contact_engine(ContactEmbeddingPredictor),
    _contact_engine(ContactAttentionPredictor),
    _contact_engine(ContactJacobianPredictor),
)


def resolve_engine(
    config: EvaluationConfig, engines: Sequence[EngineSpec] = ENGINES
) -> TaskDefinition | None:
    """Return the task that runs *config*, or ``None`` for the legacy pipeline.

    Raises if more than one engine claims the config, since the choice would then
    depend on registration order.
    """
    matches = [spec for spec in engines if spec.supports(config)]
    if len(matches) > 1:
        names = ", ".join(spec.name for spec in matches)
        raise ValueError(f"Several engines claim this job: {names}.")
    return matches[0].build(config) if matches else None
