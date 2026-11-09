"""Sequence-level regression: one continuous target per sequence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification

from modules.evaluate.src.contracts.artifacts import PredictOutput
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.metrics.bootstrap import mean_per_example_scores
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
    dataset_baselines = frozenset({"train_mean", "composition", "mmseqs_knn"})
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

    def prepare_scoring(
        self, predictions: PredictOutput, config: ScoreConfig
    ) -> tuple[FlatMetricsTask, PredictOutput]:
        group_column = config.group_column
        if group_column is None:
            return self, predictions
        frame = predictions.predictions
        if group_column not in frame.columns:
            raise ValueError(
                f"Grouped regression scoring requires '{group_column}' in predictions; "
                "rerun preparation and prediction with the annotated data."
            )
        cluster_ids = frame[group_column]
        if cluster_ids.null_count() or (
            cluster_ids.dtype.is_float() and not cluster_ids.is_finite().all()
        ):
            raise ValueError("Metric group IDs must be non-null and finite.")
        if frame["id"].n_unique() != frame.height:
            raise ValueError(
                "Grouped regression predictions require unique original IDs."
            )

        def grouped(rows: pl.DataFrame) -> pl.DataFrame:
            return (
                rows.group_by(group_column, maintain_order=True)
                .agg("prediction", "label")
                .rename({group_column: "id"})
            )

        controls = {}
        for name, control in predictions.controls.items():
            if (
                control.height != frame.height
                or control["id"].n_unique() != control.height
                or set(control["id"]) != set(frame["id"])
            ):
                raise ValueError(
                    f"Grouped regression control '{name}' has mismatched IDs."
                )
            aligned = frame.select("id", group_column, "label").join(
                control.select("id", "prediction"),
                on="id",
                how="left",
                validate="1:1",
                maintain_order="left",
            )
            controls[name] = grouped(aligned)
        return _ClusteredRegressionScoring(), replace(
            predictions, predictions=grouped(frame), controls=controls
        )

    def compute_grouped_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        group_ids: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
    ) -> dict[str, float]:
        if len(predictions) != len(labels) or len(predictions) != len(group_ids):
            raise ValueError(
                "Grouped regression predictions, labels, and group IDs must align."
            )
        if len(predictions) == 0:
            raise ValueError("Cannot compute grouped metrics from empty predictions.")
        frame = pl.DataFrame(
            {
                "group": group_ids,
                "prediction": predictions,
                "label": labels,
            }
        )
        grouped = frame.group_by("group", maintain_order=True).agg(
            "prediction", "label"
        )
        return _ClusteredRegressionScoring().compute_metrics(
            grouped["prediction"].to_list(),
            grouped["label"].to_list(),
            metric_names,
            average,
        )

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


class _ClusteredRegressionScoring(SequenceRegressionTask):
    """Equal-weight cluster metrics, with one resampling unit per cluster."""

    is_ragged = True

    def prepare_scoring(
        self, predictions: PredictOutput, config: ScoreConfig
    ) -> tuple[FlatMetricsTask, PredictOutput]:
        return self, predictions

    def compute_per_example_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> np.ndarray:
        self.validate_metric_names(metric_names)
        scores = []
        for cluster_predictions, cluster_labels in zip(
            predictions, labels, strict=True
        ):
            cluster_scores = []
            for name in metric_names:
                try:
                    score = super().compute_metrics(
                        cluster_predictions, cluster_labels, [name], average
                    )[name]
                except UndefinedMetricError:
                    score = float("nan")
                cluster_scores.append(score)
            scores.append(cluster_scores)
        return np.asarray(scores, dtype=float).reshape(
            len(predictions), len(metric_names)
        )

    def compute_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> dict[str, float]:
        return mean_per_example_scores(
            self.compute_per_example_metrics(
                predictions, labels, metric_names, average
            ),
            list(metric_names),
        )

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        lengths = [len(cluster) for cluster in labels]
        shuffled = rng.permutation(np.concatenate(labels))
        return [part.tolist() for part in np.split(shuffled, np.cumsum(lengths)[:-1])]
