"""Sequence-level classification: one class label per sequence."""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task


@register_task
class SequenceClassificationTask(TaskHandler):
    """One class label per sequence.

    The default metric set is accuracy, F1, MCC, and ROC-AUC. Precision and
    recall are available through task-specific metric extensions when needed.
    MCC is included because it is informative for imbalanced classes.
    """

    name = "sequence_classification"
    auto_model_class = AutoModelForSequenceClassification
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"scalar"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "mcc": metrics.mcc,
        "roc_auc": metrics.roc_auc,
    }
    metrics_requiring_probabilities = frozenset({"roc_auc"})
    default_metric_names = ("f1", "accuracy", "mcc", "roc_auc")

    def validate_num_labels(
        self,
        dataset: Any,
        label_column: str,
        num_labels: int,
        dataset_name: str,
    ) -> None:
        self._validate_categorical_num_labels(
            dataset, label_column, num_labels, dataset_name
        )
