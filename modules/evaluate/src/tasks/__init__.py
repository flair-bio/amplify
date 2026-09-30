"""Task-type handlers for the evaluation pipeline.

Importing this package registers every built-in task type. Always import
``get_task``/``available_tasks`` from here (not from ``.registry``) so the
handler modules are guaranteed to have been imported.
"""

from modules.evaluate.src.tasks.base import TaskHandler
from modules.evaluate.src.tasks.registry import (
    available_tasks,
    get_task,
    register_task,
)

# Imported for their @register_task side effect.
from modules.evaluate.src.tasks import (  # noqa: F401  (isort: skip)
    categorical_jacobian,
    contact_prediction,
    pseudo_perplexity,
    sequence_classification,
    sequence_regression,
    token_classification,
)

__all__ = [
    "TaskHandler",
    "TaskType",
    "available_tasks",
    "get_task",
    "register_task",
]

# Valid values are whatever is registered above, so this is a plain alias rather
# than a Literal; EvaluationWorkspaceConfig validates against the registry.
TaskType = str
