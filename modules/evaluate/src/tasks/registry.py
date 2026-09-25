"""Registry mapping ``workspace.task_type`` values to their handler."""

from __future__ import annotations

from modules.evaluate.src.tasks.base import TaskHandler

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


def get_task(task_type: str) -> TaskHandler:
    """Instantiate the handler registered for ``task_type``."""
    try:
        handler_cls = _TASK_REGISTRY[task_type]
    except KeyError:
        raise KeyError(
            f"Unknown task_type '{task_type}'. Available: {available_tasks()}."
        ) from None
    return handler_cls()
