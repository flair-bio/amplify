"""Task semantics, loaded only when selected."""

from modules.evaluate.src.tasks.base import FlatTask
from modules.evaluate.src.tasks.registry import available_tasks, get_task

__all__ = [
    "FlatTask",
    "TaskType",
    "available_tasks",
    "get_task",
]

TaskType = str
