"""Model-independent constant predictors fitted on training targets."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import polars as pl
import torch

from modules.evaluate.src.contracts.config import BaselineName
from modules.evaluate.src.tasks import get_task


def build_sequence_baseline(
    name: BaselineName,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_labels: int | None,
) -> pl.DataFrame:
    """Build train-only sequence-level constant prediction rows."""
    labels = [row["label"] for row in train_rows]
    probabilities = None
    prediction: int | float
    if name == "majority_class":
        frequencies: dict[int, int] = defaultdict(int)
        for label in labels:
            frequencies[label] += 1
        counts = [frequencies[index] for index in range(num_labels or 0)]
        prediction = max(range(len(counts)), key=counts.__getitem__)
        probabilities = [count / len(labels) for count in counts]
    else:
        prediction = math.fsum(labels) / len(labels)
    data = {
        "id": [row["id"] for row in test_rows],
        "label": [row["label"] for row in test_rows],
        "prediction": [prediction] * len(test_rows),
    }
    if probabilities is not None:
        data["probability"] = [probabilities] * len(test_rows)
    return pl.DataFrame(data)


def build_prediction_baseline(
    name: BaselineName,
    task_type: str,
    dataset_name: str,
    num_labels: int | None,
    predictions: pl.DataFrame,
    train_labels: torch.Tensor,
) -> pl.DataFrame:
    """Build sequence-level or token-level constant baseline rows."""
    if name == "majority_class":
        if task_type == "sequence_regression":
            raise ValueError(
                "baseline='majority_class' is only defined for classification "
                "tasks; use the train-mean regression baseline instead."
            )
        class_count = get_task(task_type).resolve_num_labels(num_labels, dataset_name)
        counts = torch.bincount(train_labels.to(torch.long), minlength=class_count)
        majority_class = int(counts.argmax().item())
        class_probabilities = (counts / counts.sum()).tolist()
        baseline_predictions = [
            (
                [majority_class] * len(prediction)
                if isinstance(prediction, list)
                else majority_class
            )
            for prediction in predictions["prediction"].to_list()
        ]
        data: dict[str, list[Any]] = {
            "id": predictions["id"].to_list(),
            "prediction": baseline_predictions,
        }
        if "probability" in predictions.columns:
            data["probability"] = [
                (
                    [class_probabilities] * len(prediction)
                    if isinstance(prediction, list)
                    else class_probabilities
                )
                for prediction in baseline_predictions
            ]
        return pl.DataFrame(data)
    if task_type != "sequence_regression":
        raise ValueError(
            "baseline='train_mean' is only defined for sequence_regression tasks; "
            "use a majority_class baseline for classification tasks."
        )
    train_mean = train_labels.to(torch.float32).mean().item()
    return pl.DataFrame(
        {
            "id": predictions["id"].to_list(),
            "prediction": [train_mean] * predictions.height,
        }
    )
