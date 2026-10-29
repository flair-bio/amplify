"""Task-type handlers for the evaluation pipeline.

Importing this package registers every built-in task type. Always import
``get_task``/``available_tasks`` from here (not from ``.registry``) so the
handler modules are guaranteed to have been imported.
"""

# Imported for their @register_task side effect.
from modules.evaluate.src.tasks import (
    calibration as calibration,
)
from modules.evaluate.src.tasks import (
    contact_prediction as contact_prediction,
)
from modules.evaluate.src.tasks import (
    pseudo_perplexity as pseudo_perplexity,
)
from modules.evaluate.src.tasks import (
    sequence_classification as sequence_classification,
)
from modules.evaluate.src.tasks import (
    sequence_multilabel_classification as sequence_multilabel_classification,
)
from modules.evaluate.src.tasks import (
    sequence_regression as sequence_regression,
)
from modules.evaluate.src.tasks import (
    token_classification as token_classification,
)
from modules.evaluate.src.tasks.base import TaskHandler
from modules.evaluate.src.tasks.registry import available_tasks, get_task, register_task

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
