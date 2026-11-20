"""Lazy, explicit built-in task lookup with per-run classification vocabularies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from modules.evaluate.src.tasks.base import FlatTask

_TASKS = {
    "calibration": "CalibrationTask",
    "pseudo_perplexity": "PseudoPerplexityTask",
    "sequence_classification": "SequenceClassificationTask",
    "sequence_multilabel_classification": "SequenceMultilabelClassificationTask",
    "sequence_regression": "SequenceRegressionTask",
    "token_classification": "TokenClassificationTask",
    "contact_prediction": None,
}


def available_tasks() -> list[str]:
    return sorted(_TASKS)


def get_task(task_type: str, num_labels: int | None = None) -> Any:
    """Resolve flat semantics or the independent contact protocol module."""
    if task_type not in _TASKS:
        raise KeyError(
            f"Unknown task_type '{task_type}'. Available: {available_tasks()}."
        )
    module = import_module(f"modules.evaluate.src.tasks.{task_type}")
    class_name = _TASKS[task_type]
    if class_name is None:
        return module
    task = getattr(module, class_name)()
    if num_labels is not None and task_type in {
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    }:
        count = task.resolve_num_labels(num_labels, dataset_name="")
        task.class_labels = tuple(range(count))
    return task


def get_flat_task(task_type: str, num_labels: int | None = None) -> FlatTask:
    """Resolve the flat scoring capability or reject structured tasks."""
    task = get_task(task_type, num_labels)
    if not isinstance(task, FlatTask):
        raise TypeError(f"Task '{task_type}' requires structured scoring.")
    return task
