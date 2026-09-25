"""Edge-case tests for the per-task metric dispatch."""

from __future__ import annotations

import pytest

from modules.evaluate.src.tasks import get_task


def test_mse_allows_a_single_example():
    task = get_task("sequence_regression")
    scores = task.compute_metrics(
        predictions=[2.0],
        labels=[1.0],
        metric_names=["mse"],
    )

    assert scores == {"mse": 1.0}
