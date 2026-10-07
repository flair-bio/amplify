"""Sequence-level multilabel classification: multiple independent binary
labels per sequence (e.g. GO term annotation), trained with BCE-with-logits."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task


@register_task
class SequenceMultilabelClassificationTask(TaskHandler):
    """Multiple independent binary labels per sequence.

    Each source label is a fixed-width binary indicator vector, collated as a
    ``(batch, num_labels)`` float matrix, matching what
    ``AutoModelForSequenceClassification``'s ``BCEWithLogitsLoss`` path
    (``problem_type='multi_label_classification'``) expects.
    """

    name = "sequence_multilabel_classification"
    auto_model_class = AutoModelForSequenceClassification
    label_dtype = torch.float
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"multilabel"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "multilabel_mcc": metrics.multilabel_mcc,
        "average_precision": metrics.average_precision,
        "multilabel_roc_auc": metrics.multilabel_roc_auc,
    }
    metrics_requiring_probabilities = frozenset(
        {"average_precision", "multilabel_roc_auc"}
    )
    default_metric_names = ("f1", "multilabel_mcc", "average_precision")

    def model_kwargs(self) -> dict[str, Any]:
        return {"problem_type": "multi_label_classification"}

    def validate_num_labels(
        self,
        dataset: Any,
        label_column: str,
        num_labels: int,
        dataset_name: str,
    ) -> None:
        for split_name, split in dataset.items():
            for row_index, label in enumerate(split[label_column]):
                vector = np.asarray(label)
                if vector.shape != (num_labels,):
                    raise ValueError(
                        f"task_type='{self.name}' dataset '{dataset_name}' split "
                        f"'{split_name}' row {row_index} must contain a multilabel "
                        f"vector of length {num_labels}, got shape {vector.shape}."
                    )
                if not np.isin(vector, (0, 1)).all():
                    raise ValueError(
                        f"task_type='{self.name}' dataset '{dataset_name}' split "
                        f"'{split_name}' row {row_index} must contain only binary "
                        "values (0 or 1)."
                    )

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return (torch.sigmoid(logits) > 0.5).to(dtype=torch.long)

    def predicted_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
