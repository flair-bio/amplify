"""Edge-case tests for the per-task metric dispatch."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from modules.evaluate.src.metrics.metrics import UndefinedMetricError, _binary_roc_auc, _mcc_score, average_precision, multilabel_mcc, roc_auc
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


@pytest.mark.parametrize("score_type", ["continuous", "tied", "constant"])
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_roc_auc_matches_sklearn(score_type: str, weighted: bool) -> None:
    rng = np.random.default_rng(710019)
    labels = np.repeat([False, True], [53, 7])
    if score_type == "continuous":
        probabilities = rng.uniform(size=len(labels))
    elif score_type == "tied":
        probabilities = rng.integers(0, 4, size=len(labels)).astype(float)
    else:
        probabilities = np.ones(len(labels))
    weights = rng.uniform(0.01, 1.0, size=len(labels)) if weighted else None

    expected = roc_auc_score(labels, probabilities, sample_weight=weights)
    assert _binary_roc_auc(probabilities, labels, weights) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize("score_type", ["continuous", "tied", "constant"])
@pytest.mark.parametrize("resample", [False, True])
def test_multiclass_auc_matches_sklearn(
    average: str, score_type: str, resample: bool
) -> None:
    rng = np.random.default_rng(710019)
    labels = np.repeat([0, 2, 4], [36, 18, 6])
    shape = (len(labels), 6)

    if score_type == "continuous":
        probabilities = rng.uniform(0.1, 1.0, size=shape)
    elif score_type == "tied":
        probabilities = rng.integers(1, 4, size=shape).astype(float)
    else:
        probabilities = np.ones(shape)

    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if resample:
        indices = rng.integers(0, len(labels), size=len(labels))
        labels = labels[indices]
        probabilities = probabilities[indices]

    observed_classes = np.unique(labels)
    if len(observed_classes) == 2:
        expected = roc_auc_score(
            labels, probabilities[:, observed_classes[1]], average=average
        )
    else:
        observed_probs = probabilities[:, observed_classes]
        observed_probs /= observed_probs.sum(axis=1, keepdims=True)
        expected = roc_auc_score(
            labels,
            observed_probs,
            multi_class="ovo",
            average=average,
            labels=observed_classes,
        )

    assert roc_auc(probabilities, labels, average) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("average", ["micro", "samples"])
def test_multiclass_auc_rejects_unsupported_average(average: str) -> None:
    with pytest.raises(ValueError, match="average"):
        roc_auc(np.eye(3), [0, 1, 2], average)


@pytest.mark.parametrize("invalid_probability", [np.nan, np.inf])
def test_multiclass_auc_rejects_nonfinite_probabilities(
    invalid_probability: float,
) -> None:
    probabilities = np.eye(3)
    probabilities[0, 0] = invalid_probability
    with np.errstate(invalid="ignore"), pytest.raises(ValueError):
        roc_auc(probabilities, [0, 1, 2], "weighted")


@pytest.mark.filterwarnings("ignore:No positive class found in y_true:UserWarning")
@pytest.mark.parametrize("average", ["macro", "weighted", "micro", "samples"])
@pytest.mark.parametrize("score_type", ["continuous", "tied", "constant"])
@pytest.mark.parametrize("label_type", ["mixed", "all_negative", "all_positive"])
@pytest.mark.parametrize("resample", [False, True])
def test_average_precision_matches_sklearn(
    average: str, score_type: str, label_type: str, resample: bool
) -> None:
    rng = np.random.default_rng(710019)
    labels = (rng.random((30, 5)) < [0.0, 0.05, 0.25, 0.5, 1.0]).astype(np.float32)
    if label_type == "all_negative":
        labels.fill(0)
    elif label_type == "all_positive":
        labels.fill(1)

    if score_type == "continuous":
        probabilities = rng.uniform(size=labels.shape)
    elif score_type == "tied":
        probabilities = rng.integers(0, 4, size=labels.shape).astype(float) / 3
    else:
        probabilities = np.full(labels.shape, 0.5)

    if resample:
        indices = rng.integers(0, len(labels), size=len(labels))
        labels = labels[indices]
        probabilities = probabilities[indices]

    expected = average_precision_score(labels, probabilities, average=average)
    assert average_precision(probabilities, labels, average) == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


@pytest.mark.parametrize("average", ["macro", "weighted", "micro", "samples"])
@pytest.mark.parametrize("column_vector", [False, True])
def test_binary_average_precision_matches_sklearn(
    average: str, column_vector: bool
) -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.3, 0.3, 0.9, 0.7])
    if column_vector:
        labels = labels[:, None]
        probabilities = probabilities[:, None]

    expected = average_precision_score(labels, probabilities, average=average)
    assert average_precision(probabilities, labels, average) == pytest.approx(expected)


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize(
    "probabilities, labels",
    [
        ([[np.nan, 0.5], [0.3, 0.8]], [[1, 0], [0, 1]]),
        ([[np.inf, 0.5], [0.3, 0.8]], [[1, 0], [0, 1]]),
        ([[0.7, 0.3]], [[1, 0], [0, 1]]),
        ([[0.7, 0.3], [0.2, 0.8]], [[1, np.nan], [0, 1]]),
        ([[0.7, 0.3], [0.2, 0.8]], [[0.5, 0.5], [0.5, 0.5]]),
        (np.empty((0, 2)), np.empty((0, 2))),
    ],
)
def test_average_precision_rejects_invalid_inputs(
    probabilities, labels, average: str
) -> None:
    with pytest.raises(ValueError):
        average_precision(probabilities, labels, average)


def _reference_multilabel_mcc(predictions, labels, average):
    """The original per-label/per-example MCC calculation."""
    if average == "micro":
        return _mcc_score(labels.ravel(), predictions.ravel())
    if average == "samples":
        return float(
            np.mean(
                [
                    _mcc_score(label_row, prediction_row)
                    for label_row, prediction_row in zip(labels, predictions)
                ]
            )
        )
    label_scores = np.asarray(
        [
            _mcc_score(labels[:, column], predictions[:, column])
            for column in range(labels.shape[1])
        ]
    )
    support = labels.sum(axis=0, dtype=float)
    if average == "macro" or support.sum() == 0:
        return float(np.mean(label_scores))
    return float(np.average(label_scores, weights=support))


@pytest.mark.parametrize("average", ["macro", "weighted", "micro", "samples"])
@pytest.mark.parametrize("dtype", [bool, np.int8, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 5), (37, 1), (60, 5)])
@pytest.mark.parametrize("resample", [False, True])
def test_multilabel_mcc_matches_original(average, dtype, shape, resample):
    rng = np.random.default_rng(710019)
    labels = (rng.random(shape) < rng.uniform(0.1, 0.9, shape[1])).astype(dtype)
    predictions = (rng.random(shape) < rng.uniform(0.1, 0.9, shape[1])).astype(dtype)
    if shape[1] > 1:
        labels[:, 0] = 0
        predictions[:, -1] = 1
    if resample:
        indices = rng.integers(0, len(labels), size=len(labels))
        labels = labels[indices]
        predictions = predictions[indices]

    expected = _reference_multilabel_mcc(predictions, labels, average)
    assert multilabel_mcc(predictions, labels, average) == expected


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize("num_examples", [1, 2, 3, 4])
def test_multilabel_mcc_exhaustive_binary_columns(average, num_examples):
    vectors = list(product((0, 1), repeat=num_examples))
    for label_values, prediction_values in product(vectors, repeat=2):
        labels = np.asarray(label_values)[:, None]
        predictions = np.asarray(prediction_values)[:, None]
        expected = _reference_multilabel_mcc(predictions, labels, average)
        assert multilabel_mcc(predictions, labels, average) == expected


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize(
    "case",
    [
        "perfect",
        "inverse",
        "all_negative_labels",
        "all_positive_labels",
        "all_negative_predictions",
        "all_positive_predictions",
    ],
)
def test_multilabel_mcc_degenerate_columns_match_original(average, case):
    labels = np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
    predictions = labels.copy()
    if case == "inverse":
        predictions = 1 - labels
    elif case == "all_negative_labels":
        labels.fill(0)
    elif case == "all_positive_labels":
        labels.fill(1)
    elif case == "all_negative_predictions":
        predictions.fill(0)
    elif case == "all_positive_predictions":
        predictions.fill(1)

    expected = _reference_multilabel_mcc(predictions, labels, average)
    with np.errstate(divide="raise", invalid="raise"):
        assert multilabel_mcc(predictions, labels, average) == expected


@pytest.mark.parametrize("average", ["macro", "weighted", "micro", "samples"])
@pytest.mark.parametrize("encoding", [(-1, 1), (0, 2), (1, 2)])
def test_multilabel_mcc_other_encodings_match_original(average, encoding):
    labels = np.asarray(encoding)[[[0, 1], [1, 0], [1, 1]]]
    predictions = np.asarray(encoding)[[[0, 1], [1, 0], [0, 0]]]
    expected = _reference_multilabel_mcc(predictions, labels, average)
    assert multilabel_mcc(predictions, labels, average) == expected


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize(
    "predictions, labels",
    [
        ([0, 1], [[0, 1]]),
        ([[0, 1]], [0, 1]),
        ([[0, 1]], [[0], [1]]),
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.empty((2, 0)), np.empty((2, 0))),
    ],
)
def test_multilabel_mcc_rejects_invalid_shapes(predictions, labels, average):
    with pytest.raises(ValueError, match="Multilabel MCC"):
        multilabel_mcc(predictions, labels, average)
