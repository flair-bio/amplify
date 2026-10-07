"""
Shared metric callables reused by the task handlers.

Each callable takes the uniform ``(values, labels, average)`` signature that
:meth:`modules.evaluate.src.tasks.base.TaskHandler.compute_metrics` dispatches
with; ``values`` is the predictions except for probability-based metrics such
as ``roc_auc``. Which of these a task actually supports is declared by that
task's handler in ``modules/evaluate/src/tasks/``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


class UndefinedMetricError(ValueError):
    """A statistic is undefined for this sample, not malformed input."""


def accuracy(predictions: Any, labels: Any, average: str) -> float:
    return float(accuracy_score(labels, predictions))


def f1(predictions: Any, labels: Any, average: str) -> float:
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def precision(predictions: Any, labels: Any, average: str) -> float:
    return float(precision_score(labels, predictions, average=average, zero_division=0))


def recall(predictions: Any, labels: Any, average: str) -> float:
    return float(recall_score(labels, predictions, average=average, zero_division=0))


def _mcc_score(labels: Any, predictions: Any) -> float:
    labels_arr = np.asarray(labels)
    predictions_arr = np.asarray(predictions)
    if np.unique(labels_arr).size < 2 or np.unique(predictions_arr).size < 2:
        return 0.0
    return float(matthews_corrcoef(labels_arr, predictions_arr))


def mcc(predictions: Any, labels: Any, average: str) -> float:
    """Matthews correlation coefficient. Ignores ``average``: the same
    formulation covers both binary and multi-class."""
    return _mcc_score(labels, predictions)


def multilabel_mcc(predictions: Any, labels: Any, average: str) -> float:
    """MCC aggregated across labels or examples using the requested average.

    ``weighted`` averages per-label MCC by positive-label support; when there
    are no positive labels, it falls back to the unweighted label mean.
    """
    predictions_arr = np.asarray(predictions)
    labels_arr = np.asarray(labels)
    if predictions_arr.ndim != 2 or labels_arr.ndim != 2:
        raise ValueError("Multilabel MCC requires 2D prediction and label matrices.")
    if predictions_arr.shape != labels_arr.shape:
        raise ValueError("Multilabel MCC prediction and label shapes must match.")
    if labels_arr.shape[0] == 0 or labels_arr.shape[1] == 0:
        raise ValueError("Multilabel MCC requires at least one example and label.")

    if average == "micro":
        return _mcc_score(labels_arr.ravel(), predictions_arr.ravel())
    if average == "samples":
        return float(
            np.mean(
                [
                    _mcc_score(label_row, prediction_row)
                    for label_row, prediction_row in zip(labels_arr, predictions_arr)
                ]
            )
        )
    if average not in {"macro", "weighted"}:
        raise ValueError(
            "Multilabel MCC average must be one of 'micro', 'macro', "
            "'weighted', or 'samples'."
        )

    label_scores = np.asarray(
        [
            _mcc_score(labels_arr[:, index], predictions_arr[:, index])
            for index in range(labels_arr.shape[1])
        ]
    )
    if average == "macro":
        return float(np.mean(label_scores))

    support = labels_arr.sum(axis=0, dtype=float)
    if support.sum() == 0:
        return float(np.mean(label_scores))
    return float(np.average(label_scores, weights=support))


def mse(predictions: Any, labels: Any, average: str) -> float:
    return float(mean_squared_error(labels, predictions))


def mae(predictions: Any, labels: Any, average: str) -> float:
    return float(mean_absolute_error(labels, predictions))


def mean_log_prob(predictions: Any, labels: Any, average: str) -> float:
    return float(np.mean(predictions))


def pseudo_perplexity(predictions: Any, labels: Any, average: str) -> float:
    return float(np.exp(-np.mean(predictions)))


def r2(predictions: Any, labels: Any, average: str) -> float:
    return float(r2_score(labels, predictions))


def pearson(predictions: Any, labels: Any, average: str) -> float:
    return float(pearsonr(predictions, labels)[0])


def spearman(predictions: Any, labels: Any, average: str) -> float:
    return float(spearmanr(predictions, labels)[0])


def roc_auc(probabilities: Sequence, labels: Sequence, average: str) -> float:
    """ROC-AUC from per-class probabilities, binary or multi-class (one-vs-one).

    Supports binary and multiclass probability matrices using the same metric
    contract as the other task handlers.
    """
    probs = np.asarray(probabilities, dtype=float)
    labels_arr = np.asarray(labels)
    if probs.ndim != 2:
        raise ValueError(
            "ROC-AUC probabilities must have shape (n_examples, n_classes)."
        )
    if probs.shape[0] != labels_arr.size:
        raise ValueError("ROC-AUC probabilities and labels must have equal lengths.")
    num_classes = probs.shape[1]
    observed_classes = np.unique(labels_arr)
    if np.any(observed_classes < 0) or np.any(observed_classes >= num_classes):
        raise ValueError("ROC-AUC labels must index columns of the probability matrix.")
    if len(observed_classes) < 2:
        raise UndefinedMetricError("ROC-AUC requires at least two observed classes.")
    if len(observed_classes) == 2:
        return float(
            roc_auc_score(labels_arr, probs[:, observed_classes[1]], average=average)
        )
    observed_probs = probs[:, observed_classes]
    observed_probs = observed_probs / observed_probs.sum(axis=1, keepdims=True)
    return float(
        roc_auc_score(
            labels_arr,
            observed_probs,
            multi_class="ovo",
            average=average,
            labels=observed_classes,
        )
    )


def average_precision(probabilities: Sequence, labels: Sequence, average: str) -> float:
    """Average precision from per-label probabilities, for multilabel
    indicator matrices (``average`` in {'micro', 'macro', 'samples'})."""
    return float(
        average_precision_score(
            np.asarray(labels), np.asarray(probabilities), average=average
        )
    )


def multilabel_roc_auc(
    probabilities: Sequence, labels: Sequence, average: str
) -> float:
    """ROC-AUC over a multilabel indicator matrix.

    'micro'/'samples' delegate straight to sklearn, which aggregates over
    predictions directly. 'macro' is computed by hand instead, skipping any
    label column with only one class present in this split (undefined for
    that column) rather than letting sklearn raise for the whole metric.
    """
    probs = np.asarray(probabilities, dtype=float)
    labels_arr = np.asarray(labels)
    if labels_arr.ndim != 2:
        raise ValueError("Multilabel ROC-AUC requires a 2D label indicator matrix.")
    if average != "macro":
        return float(roc_auc_score(labels_arr, probs, average=average))
    defined_columns = [
        j for j in range(labels_arr.shape[1]) if len(np.unique(labels_arr[:, j])) > 1
    ]
    if not defined_columns:
        raise UndefinedMetricError(
            "Multilabel ROC-AUC is undefined: every label column has only one "
            "class in this split."
        )
    return float(
        np.mean([roc_auc_score(labels_arr[:, j], probs[:, j]) for j in defined_columns])
    )
