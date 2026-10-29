"""Sequence-level regression: one continuous target per sequence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.metrics.metrics import UndefinedMetricError
from modules.evaluate.src.tasks.base import FlatMetricsTask, MetricFn
from modules.evaluate.src.tasks.registry import register_task

# Undefined on a single example, so they are rejected up front rather than
# returning NaN.
_METRICS_REQUIRING_TWO_EXAMPLES = frozenset({"r2", "pearsonr", "spearmanr"})
_CORRELATION_METRICS = frozenset({"pearsonr", "spearmanr"})


@register_task
class SequenceRegressionTask(FlatMetricsTask):
    """One continuous target per sequence with regression metrics."""

    supports_layerwise = True
    dataset_baselines = frozenset({"train_mean"})
    prediction_baselines = frozenset({"train_mean"})

    name = "sequence_regression"
    # Same wrapper class as classification; problem_type switches the loss.
    auto_model_class = AutoModelForSequenceClassification
    label_dtype = torch.float
    accepted_target_types = frozenset({"continuous"})
    accepted_label_formats = frozenset({"scalar"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "r2": metrics.r2,
        "pearsonr": metrics.pearson,
        "spearmanr": metrics.spearman,
    }
    default_metric_names = ("mse", "mae", "r2", "pearsonr", "spearmanr")

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        return torch.tensor([float(label) for label in labels], dtype=self.label_dtype)

    def model_kwargs(self) -> dict[str, Any]:
        # HF uses problem_type to switch the head/loss to a regression
        # objective even though the wrapper class is the same.
        return {"problem_type": "regression"}

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        return 1

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return logits.squeeze(-1)

    def compute_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> dict[str, float]:
        self.validate_metric_names(metric_names)
        requested = set(metric_names)
        if requested & _METRICS_REQUIRING_TWO_EXAMPLES and len(predictions) < 2:
            raise UndefinedMetricError(
                "R2 and correlation metrics require at least two examples."
            )
        if requested & _CORRELATION_METRICS and (
            np.ptp(np.asarray(predictions)) == 0 or np.ptp(np.asarray(labels)) == 0
        ):
            raise UndefinedMetricError(
                "Pearson and Spearman correlations are undefined when "
                "predictions or labels are constant."
            )
        return super().compute_metrics(
            predictions, labels, metric_names, average, probabilities
        )
