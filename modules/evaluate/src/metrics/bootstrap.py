"""Pure statistical helpers behind ScoreStep's bootstrap confidence intervals
and trained-vs-control comparisons: array conversion, per-resample/full-set
metric scoring, the paired permutation test, and multiple-comparisons
correction. Kept out of score.py since none of these need ScoreStep's own
state (config/workspace/IO) -- each takes everything it needs as a plain
argument, so they're independently testable and reusable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from modules.evaluate.src.contracts.artifacts import (
    ConfidenceInterval,
    PredictOutput,
)
from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import FlatMetricsTask

logger = logging.getLogger(__name__)


def as_indexable_array(values: list[Any], ragged: bool) -> np.ndarray:
    """Convert a per-example column to a 1D NumPy array suitable for
    fancy indexing (``arr[idx]``) during bootstrap resampling.

    ``token_classification`` predictions/labels are one variable-length
    sequence per example (ragged), which ``np.asarray`` cannot pack into
    a rectangular array; those are stored as a 1D object array instead,
    preserving per-example fancy indexing while keeping each example's
    original sequence intact.
    """
    if ragged:
        arr = np.empty(len(values), dtype=object)
        arr[:] = values
        return arr
    return np.asarray(values)


def id_groups(ids: list[Any] | None) -> np.ndarray | None:
    """Map each row to its source-ID group, or None when IDs are absent or unique.

    Cached token predictions repeat a sequence's ID on every residue row;
    resampling whole groups keeps correlated residues of one protein together.
    """
    if ids is None:
        return None
    index: dict[Any, int] = {}
    groups = np.fromiter(
        (index.setdefault(row_id, len(index)) for row_id in ids),
        dtype=np.int64,
        count=len(ids),
    )
    return None if len(index) == len(ids) else groups


def bootstrap_indices(
    n_rows: int, groups: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Row indices for one bootstrap resample, drawing whole ID groups."""
    if groups is None:
        return rng.integers(0, n_rows, size=n_rows)
    n_groups = int(groups.max()) + 1
    counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
    return np.repeat(np.arange(n_rows), counts[groups])


def token_confusion_matrices(
    task: FlatMetricsTask,
    conditions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    groups: np.ndarray | None,
    metric_names: list[str],
    average: str,
) -> dict[str, np.ndarray] | None:
    """Prepare integer [protein, actual class, predicted class] counts once.

    All conditions share a class vocabulary so matrices can be swapped even
    when a control changes labels or predicts classes absent from the truth.
    Custom metrics and unsupported aggregation modes retain the ordinary path.
    """
    if (
        task.name != "token_classification"
        or average not in {"micro", "macro", "weighted"}
        or any(
            name not in {"accuracy", "f1", "precision", "recall", "mcc"}
            or task.metrics[name] is not getattr(metrics, name)
            for name in metric_names
        )
    ):
        return None

    flattened = {}
    for name, (predictions, labels, _) in conditions.items():
        pred_lengths = np.fromiter(map(len, predictions), dtype=np.int64)
        label_lengths = np.fromiter(map(len, labels), dtype=np.int64)
        if not np.array_equal(pred_lengths, label_lengths):
            raise ValueError("Token prediction and label lengths must match per row.")
        flat_preds = np.asarray(task.flatten_for_metrics(predictions.tolist()))
        flat_labels = np.asarray(task.flatten_for_metrics(labels.tolist()))
        task.validate_metric_inputs(flat_preds, flat_labels)
        if (
            not len(flat_preds)
            or flat_preds.dtype.kind not in "iu"
            or flat_labels.dtype.kind not in "iu"
        ):
            return None
        flattened[name] = (flat_preds, flat_labels, pred_lengths)

    classes = (
        np.asarray(task.class_labels)
        if task.class_labels is not None
        else np.unique(
            np.concatenate(
                [
                    array
                    for preds, labels, _ in flattened.values()
                    for array in (preds, labels)
                ]
            )
        )
    )
    n_classes = len(classes)
    first_predictions = next(iter(conditions.values()))[0]
    row_groups = np.arange(len(first_predictions)) if groups is None else groups
    n_groups = int(row_groups.max()) + 1
    matrices = {}
    for name, (predictions, labels, lengths) in flattened.items():
        if len(lengths) != len(row_groups):
            raise ValueError("Conditions must have the same number of token rows.")
        token_groups = np.repeat(row_groups, lengths)
        indices = (
            token_groups * n_classes**2
            + np.searchsorted(classes, labels) * n_classes
            + np.searchsorted(classes, predictions)
        )
        matrices[name] = np.bincount(
            indices, minlength=n_groups * n_classes**2
        ).reshape(n_groups, n_classes, n_classes)
    return matrices


def confusion_scores(
    matrix: np.ndarray,
    metric_names: list[str],
    average: str,
    *,
    include_absent: bool = False,
) -> dict[str, float]:
    """Pooled metrics, retaining absent configured classes when requested."""
    actual = matrix.sum(axis=1, dtype=float)
    predicted = matrix.sum(axis=0, dtype=float)
    total = actual.sum()
    if total == 0:
        return dict.fromkeys(metric_names, float("nan"))
    present = (
        np.ones(len(actual), dtype=bool)
        if include_absent
        else (actual + predicted) != 0
    )
    actual, predicted = actual[present], predicted[present]
    true_positive = np.diag(matrix)[present].astype(float)
    correct = true_positive.sum()
    scores = {}
    for name in metric_names:
        if name == "accuracy":
            value = correct / total
        elif name == "mcc":
            covariance = correct * total - np.dot(actual, predicted)
            denominator = (total**2 - np.dot(actual, actual)) * (
                total**2 - np.dot(predicted, predicted)
            )
            value = covariance / np.sqrt(denominator) if denominator != 0 else 0.0
        elif average == "micro":
            value = correct / total
        else:
            numerator = 2 * true_positive if name == "f1" else true_positive
            denominator = (
                actual + predicted
                if name == "f1"
                else predicted
                if name == "precision"
                else actual
            )
            per_class = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator != 0,
            )
            value = np.average(
                per_class, weights=actual if average == "weighted" else None
            )
        scores[name] = float(value)
    return scores


def bootstrap_confusion_scores(
    matrices: dict[str, np.ndarray],
    metric_names: list[str],
    average: str,
    rng: np.random.Generator,
    *,
    include_absent: bool = False,
) -> dict[str, dict[str, float]]:
    """Draw the same protein multiplicities as bootstrap_indices, without rows."""
    n_groups = len(next(iter(matrices.values())))
    counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
    return {
        condition: confusion_scores(
            np.einsum("g,gij->ij", counts, values),
            metric_names,
            average,
            include_absent=include_absent,
        )
        for condition, values in matrices.items()
    }


def mean_per_example_scores(
    per_example_scores: np.ndarray, metric_names: list[str]
) -> dict[str, float]:
    """Average defined per-example scores in metric_names column order."""
    scores: dict[str, float] = {}
    for column, name in enumerate(metric_names):
        values = per_example_scores[:, column]
        values = values[~np.isnan(values)]
        scores[name] = float(np.mean(values)) if len(values) else float("nan")
    return scores


def full_scores(
    task: FlatMetricsTask,
    predictions: Any,
    labels: Any,
    probabilities: Any | None,
    metric_names: list[str],
    average: str,
    *,
    per_example_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute every requested metric on a set of per-example predictions/labels.

    Each metric is scored independently so a metric that is undefined on this
    data (e.g. roc_auc when a class is absent) is recorded as NaN rather than
    discarding the other, unaffected metrics. An unknown metric name still
    raises. When per_example_scores is supplied, average its columns instead
    of recomputing metrics from predictions and labels.
    """
    if per_example_scores is not None:
        task.validate_metric_names(metric_names)
        return mean_per_example_scores(per_example_scores, metric_names)

    flat_predictions = task.flatten_for_metrics(predictions)
    flat_labels = task.flatten_for_metrics(labels)
    if task.name in {
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    }:
        # Reuse the arrays instead of converting Python lists for every metric.
        flat_predictions = np.asarray(flat_predictions)
        flat_labels = np.asarray(flat_labels)
    task.validate_metric_names(metric_names)
    task.validate_metric_inputs(flat_predictions, flat_labels)

    scores: dict[str, float] = {}
    for name in metric_names:
        if name in scores:
            continue
        try:
            scores.update(
                task.compute_metrics(
                    predictions=flat_predictions,
                    labels=flat_labels,
                    metric_names=[name],
                    average=average,
                    probabilities=probabilities,
                )
            )
        except ValueError:
            scores[name] = float("nan")
    return scores


def ci_stats(values: np.ndarray, lower_q: float, upper_q: float) -> ConfidenceInterval:
    """Return the mean and percentile bounds of *values*, ignoring NaNs (all-NaN gives NaNs)."""
    values = values[~np.isnan(values)]
    if not len(values):
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, lower_q)),
        "upper": float(np.quantile(values, upper_q)),
    }


def collect_control_arrays(
    predict_result: PredictOutput,
    predictions_col: list[Any],
    labels_col: list[Any],
    probabilities_col: list[Any] | None,
    rng: np.random.Generator,
    task: FlatMetricsTask,
) -> dict[str, tuple[list[Any], list[Any], list[Any] | None]]:
    """Build (predictions, labels, probabilities) arrays for each control
    condition PredictStep was configured to produce, row-aligned to the
    trained model's predictions."""
    control_arrays: dict[str, tuple[list[Any], list[Any], list[Any] | None]] = {}

    for control, control_df in predict_result.controls.items():
        control_predictions_col = control_df["prediction"].to_list()
        control_probabilities_col = (
            control_df["probability"].to_list()
            if "probability" in control_df.columns
            else None
        )
        # Labels are unchanged for model-based controls; only predictions differ.
        control_arrays[control] = (
            control_predictions_col,
            labels_col,
            control_probabilities_col,
        )

    if "scrambled_labels" in predict_result.requested_controls:
        # No model inference needed: only the label association with the
        # trained model's predictions is destroyed.
        control_arrays["scrambled_labels"] = (
            predictions_col,
            task.scramble_labels(labels_col, rng),
            probabilities_col,
        )

    return control_arrays
