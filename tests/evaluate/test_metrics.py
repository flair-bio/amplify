"""Edge-case tests for the per-task metric dispatch."""

from __future__ import annotations

import pytest

from modules.evaluate.src.metrics.metrics import UndefinedMetricError, roc_auc
from modules.evaluate.src.tasks import get_task


def test_mse_allows_a_single_example():
    task = get_task("sequence_regression")
    scores = task.compute_metrics(
        predictions=[2.0],
        labels=[1.0],
        metric_names=["mse"],
    )

    assert scores == {"mse": 1.0}


def test_multiclass_auc_with_unobserved_class():
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8],
                     [0.7, 0.1, 0.2], [0.2, 0.1, 0.7]]
    assert roc_auc(probabilities, [0, 2, 0, 2], "weighted") == pytest.approx(1.0)
    with pytest.raises(UndefinedMetricError):
        roc_auc(probabilities, [0, 0, 0, 0], "weighted")


def test_multiclass_auc_with_three_observed_classes_and_one_missing():
    probabilities = [
        [0.8, 0.1, 0.05, 0.05],
        [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.1, 0.75, 0.05],
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ]
    assert roc_auc(probabilities, [0, 1, 2, 0, 1, 2], "weighted") == pytest.approx(1.0)
