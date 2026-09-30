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
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def accuracy(predictions: Any, labels: Any, average: str) -> float:
    return float(accuracy_score(labels, predictions))


def f1(predictions: Any, labels: Any, average: str) -> float:
    return float(f1_score(labels, predictions, average=average, zero_division=0))


def precision(predictions: Any, labels: Any, average: str) -> float:
    return float(precision_score(labels, predictions, average=average, zero_division=0))


def recall(predictions: Any, labels: Any, average: str) -> float:
    return float(recall_score(labels, predictions, average=average, zero_division=0))


def mcc(predictions: Any, labels: Any, average: str) -> float:
    """Matthews correlation coefficient. Ignores ``average``: the same
    formulation covers both binary and multi-class."""
    return float(matthews_corrcoef(labels, predictions))


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
    missing_classes = set(range(num_classes)) - set(observed_classes.tolist())
    if missing_classes:
        raise ValueError(
            "ROC-AUC is undefined because the test split lacks class(es) "
            f"{sorted(missing_classes)}."
        )
    if num_classes == 2:
        return float(roc_auc_score(labels_arr, probs[:, 1], average=average))
    return float(
        roc_auc_score(
            labels_arr,
            probs,
            multi_class="ovo",
            average=average,
            labels=np.arange(num_classes),
        )
    )
