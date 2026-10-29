"""Registry mapping ``workspace.task_type`` values to their handler."""

from __future__ import annotations

from modules.evaluate.src.tasks.base import FlatMetricsTask, TaskHandler

_TASK_REGISTRY: dict[str, type[TaskHandler]] = {}


def register_task(handler_cls: type[TaskHandler]) -> type[TaskHandler]:
    """Class decorator that makes a handler selectable via ``task_type``."""
    name = handler_cls.name
    existing = _TASK_REGISTRY.get(name)
    if existing is not None and existing is not handler_cls:
        raise ValueError(
            f"Task type '{name}' is already registered to {existing.__name__}."
        )
    _TASK_REGISTRY[name] = handler_cls
    return handler_cls


def available_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)


def get_task(task_type: str, num_labels: int | None = None) -> TaskHandler:
    """Instantiate a handler, optionally fixing its classification vocabulary."""
    try:
        handler_cls = _TASK_REGISTRY[task_type]
    except KeyError:
        raise KeyError(
            f"Unknown task_type '{task_type}'. Available: {available_tasks()}."
        ) from None
    task = handler_cls()
    if num_labels is not None and task_type in {
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    }:
        count = task.resolve_num_labels(num_labels, dataset_name="")
        task.class_labels = tuple(range(count))
    return task


def get_flat_task(task_type: str, num_labels: int | None = None) -> FlatMetricsTask:
    """Resolve the flat scoring capability or reject structured tasks."""
    task = get_task(task_type, num_labels)
    if not isinstance(task, FlatMetricsTask):
        raise TypeError(f"Task '{task_type}' requires structured scoring.")
    return task
