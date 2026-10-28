"""Tests for modules.evaluate.src.steps.score.ScoreStep.

Builds ``PreparedArtifacts``/``PredictOutput`` from the tiny, offline
``model_family`` fixtures (tests/evaluate/conftest.py), so these tests
exercise genuine metric computation, resampling, control-model construction
(random_trunk/random_head/random_both/scrambled_sequences), and label
permutation (scrambled_labels) without any Hugging Face Hub access.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    build_dataloaders,
    tokenize_dataset,
)
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.metrics.metrics import (
    _mcc_score,
    average_precision,
    multilabel_mcc,
    roc_auc,
)
from modules.evaluate.src.schemas.artifacts import (
    EvaluationMetadata,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.steps.predict import PredictConfig, PredictStep
from modules.evaluate.src.steps.score import ScoreConfig, ScoreStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.bootstrap import (
    as_indexable_array,
    benjamini_hochberg,
    bootstrap_indices,
    ci_stats,
    confusion_scores,
    full_scores,
    id_groups,
    permutation_p_values,
    summarize_controls,
    token_confusion_matrices,
)
from modules.evaluate.src.utils.io import write_parquet
from modules.evaluate.src.utils.long_format import build_long_df, upsert_long_df

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize("num_labels", [None, 8])
@pytest.mark.parametrize(
    "predictions,labels",
    [
        ([[0, 4], [], [2, 4, 0], [4]], [[0, 2], [], [2, 2, 0], [2]]),
        ([[4], [4]], [[4], [4]]),
        ([[4], [4]], [[0], [2]]),
        ([[], [4]], [[], [4]]),
    ],
)
def test_token_confusion_scores_match_residue_metrics(
    average, predictions, labels, num_labels
):
    task = get_task("token_classification", num_labels)
    names = task.default_metrics()
    arrays = (
        as_indexable_array(predictions, True),
        as_indexable_array(labels, True),
        None,
    )
    matrices = token_confusion_matrices(task, {"trained": arrays}, None, names, average)
    assert matrices is not None
    rng = np.random.default_rng(71)
    for _ in range(12):
        idx = bootstrap_indices(len(labels), None, rng)
        actual = confusion_scores(
            matrices["trained"][idx].sum(axis=0),
            names,
            average,
            include_absent=task.class_labels is not None,
        )
        expected = full_scores(
            task, arrays[0][idx], arrays[1][idx], None, names, average
        )
        np.testing.assert_allclose(
            list(actual.values()),
            list(expected.values()),
            rtol=1e-14,
            atol=1e-14,
            equal_nan=True,
        )


@pytest.mark.parametrize("num_labels", [2, 3, 8])
def test_score_and_bootstrap_keep_classes_missing_from_test_split(tmp_path, num_labels):
    task_type = "sequence_classification" if num_labels == 2 else "token_classification"
    values = [0, 0] if num_labels == 2 else [[0, 0], [0]]
    workspace = get_evaluation_workspace(
        EvaluationWorkspaceConfig(
            run_id="test",
            model_name="toy-model",
            dataset_name="toy",
            task_type=task_type,
            target_type="categorical",
            label_format="scalar" if num_labels == 2 else "per_token",
            num_labels=num_labels,
            base_path=str(tmp_path),
        )
    )
    workspace.setup_directories()
    output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name="toy-model",
            dataset_name="toy",
            task_type=task_type,
        ),
        predictions=pl.DataFrame(
            {"id": ["a", "b"], "prediction": values, "label": values}
        ),
        controls={"untrained": pl.DataFrame({"id": ["a", "b"], "prediction": values})},
        requested_controls=["untrained"],
    )
    result = ScoreStep(
        ScoreConfig(
            metrics=["f1"],
            metric_aggregation="macro",
            bootstrap_enabled=True,
            n_samples=8,
            n_permutations=8,
        )
    ).run(workspace, predict_result=output)
    assert result.scores["f1"] == pytest.approx(1 / num_labels)
    assert result.resamples["f1"].to_list() == pytest.approx([1 / num_labels] * 8)
    comparison = result.control_comparisons["untrained"]["f1"]
    assert comparison["control_mean"] == pytest.approx(1 / num_labels)
    assert comparison["diff_mean"] == 0
    assert comparison["p_value"] == 1


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize("cached_rows", [False, True])
def test_token_confusion_bootstrap_preserves_results(
    tmp_path, monkeypatch, average, cached_rows
):
    from modules.evaluate.src.steps import score as score_module
    from modules.evaluate.src.utils import bootstrap as bootstrap_module

    rng = np.random.default_rng(42)
    lengths = [2, 0, 7, 1, 3, 5, 2, 9]
    labels = [rng.choice([0, 2, 4], size=n).tolist() for n in lengths]
    predictions = [rng.choice([0, 2, 4], size=n).tolist() for n in lengths]
    # This class occurs only in a control, never in the trained condition.
    controls = [rng.choice([0, 2, 4, 6], size=n).tolist() for n in lengths]
    ids = list(range(len(lengths)))
    if cached_rows:
        ids = [i for i, row in enumerate(labels) for _ in row]
        labels = [[value] for row in labels for value in row]
        predictions = [[value] for row in predictions for value in row]
        controls = [[value] for row in controls for value in row]
    output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name="toy-model",
            dataset_name="toy_dataset",
            task_type="token_classification",
        ),
        predictions=pl.DataFrame(
            {"id": ids, "prediction": predictions, "label": labels}
        ),
        controls={"untrained": pl.DataFrame({"id": ids, "prediction": controls})},
        requested_controls=["untrained", "scrambled_labels"],
    )
    config = ScoreConfig(
        bootstrap_enabled=True,
        n_samples=24,
        n_permutations=64,
        metric_aggregation=average,
    )

    def run(name):
        workspace = get_evaluation_workspace(
            EvaluationWorkspaceConfig(
                run_id="test",
                model_name="toy-model",
                dataset_name="toy_dataset",
                task_type="token_classification",
                target_type="categorical",
                label_format="per_token",
                num_labels=7,
                base_path=str(tmp_path / name),
            )
        )
        workspace.setup_directories()
        step = ScoreStep(config)
        result = step.run(workspace, seed=123, predict_result=output)
        other_layer = PredictOutput(
            metadata=output.metadata,
            predictions=output.predictions.with_columns(
                pl.Series("prediction", controls)
            ),
        )
        comparison_path = step.compare_layers(
            workspace, {0: output, 2: other_layer}, seed=123
        )
        return result, pl.read_parquet(comparison_path)

    with monkeypatch.context() as reference:
        reference.setattr(score_module, "token_confusion_matrices", lambda *args: None)
        expected, expected_layers = run("reference")

    score_spy = MagicMock(wraps=score_module.full_scores)
    monkeypatch.setattr(score_module, "full_scores", score_spy)
    permutation_spy = MagicMock(wraps=bootstrap_module.full_scores)
    monkeypatch.setattr(bootstrap_module, "full_scores", permutation_spy)
    actual, actual_layers = run("optimized")
    assert actual.scores == expected.scores
    np.testing.assert_allclose(
        actual.resamples.to_numpy(),
        expected.resamples.to_numpy(),
        rtol=1e-14,
        atol=1e-14,
        equal_nan=True,
    )
    for name in actual.scores:
        assert actual.confidence_intervals[name] == pytest.approx(
            expected.confidence_intervals[name], abs=1e-14
        )
    for control, comparisons in actual.control_comparisons.items():
        for name, comparison in comparisons.items():
            reference = expected.control_comparisons[control][name]
            assert comparison == pytest.approx(reference, abs=1e-14)
            assert comparison["p_value"] == reference["p_value"]
            assert comparison["p_value_adjusted"] == reference["p_value_adjusted"]
    assert score_spy.call_count == 1 + len(output.requested_controls)
    permutation_spy.assert_not_called()
    assert actual.output_path.exists()
    assert actual.long_path.exists()
    for name in ["variant", "split"]:
        assert actual_layers[name].to_list() == expected_layers[name].to_list()
    for name in ["p_value", "p_value_adjusted"]:
        assert actual_layers[name].to_list() == expected_layers[name].to_list()
    np.testing.assert_allclose(
        actual_layers.drop("metric", "variant", "split").to_numpy(),
        expected_layers.drop("metric", "variant", "split").to_numpy(),
        rtol=1e-14,
        atol=1e-14,
        equal_nan=True,
    )


def test_token_confusions_fall_back_for_custom_metrics_and_samples():
    task = get_task("token_classification")
    arrays = {
        "trained": (np.array([[0]], dtype=object), np.array([[0]], dtype=object), None)
    }
    assert token_confusion_matrices(task, arrays, None, ["f1"], "samples") is None
    task.metrics = {"f1": lambda predictions, labels, average: 0.5}
    assert token_confusion_matrices(task, arrays, None, ["f1"], "macro") is None


@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize("score_type", ["continuous", "tied", "constant"])
def test_multiclass_auc_resampling_matches_sklearn(
    average: str, score_type: str
) -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 2, 4], [36, 18, 6])
    shape = (len(labels), 6)
    if score_type == "continuous":
        probabilities = rng.uniform(0.1, 1.0, size=shape)
        control_probabilities = rng.uniform(0.1, 1.0, size=shape)
    elif score_type == "tied":
        probabilities = rng.integers(1, 4, size=shape).astype(float)
        control_probabilities = rng.integers(1, 4, size=shape).astype(float)
    else:
        probabilities = np.ones(shape)
        control_probabilities = np.ones(shape)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    control_probabilities /= control_probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    control_predictions = control_probabilities.argmax(axis=1)

    def reference_auc(probabilities, labels, average):
        observed_classes = np.unique(labels)
        if len(observed_classes) == 2:
            return float(
                roc_auc_score(
                    labels, probabilities[:, observed_classes[1]], average=average
                )
            )
        observed_probs = probabilities[:, observed_classes]
        observed_probs = observed_probs / observed_probs.sum(axis=1, keepdims=True)
        return float(
            roc_auc_score(
                labels,
                observed_probs,
                multi_class="ovo",
                average=average,
                labels=observed_classes,
            )
        )

    def evaluate(metric):
        task = get_task("sequence_classification")
        task.metrics = {"roc_auc": metric}
        rng = np.random.default_rng(710019)
        resampled_scores = []
        for _ in range(20):
            indices = bootstrap_indices(len(labels), None, rng)
            scores = full_scores(
                task,
                predictions[indices],
                labels[indices],
                probabilities[indices],
                ["roc_auc"],
                average,
            )
            resampled_scores.append(scores["roc_auc"])
        observed_scores = full_scores(
            task, predictions, labels, probabilities, ["roc_auc"], average
        )
        p_values = permutation_p_values(
            task,
            observed_trained=observed_scores,
            trained_predictions=predictions,
            trained_labels=labels,
            trained_probabilities=probabilities,
            control_predictions=control_predictions,
            control_labels=labels,
            control_probabilities=control_probabilities,
            metric_names=["roc_auc"],
            n_permutations=32,
            rng=rng,
            average=average,
        )
        return ci_stats(np.asarray(resampled_scores), 0.025, 0.975), p_values

    expected_ci, expected_p_values = evaluate(reference_auc)
    actual_ci, actual_p_values = evaluate(roc_auc)
    assert actual_ci == pytest.approx(expected_ci, rel=1e-12, abs=1e-12)
    assert actual_p_values == expected_p_values


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_multilabel_average_precision_resampling_matches_sklearn(average: str) -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random((60, 12)) < rng.uniform(0.01, 0.6, 12)).astype(float)
    probabilities = rng.integers(0, 8, size=labels.shape).astype(float) / 7
    control_probabilities = rng.integers(0, 8, size=labels.shape).astype(float) / 7
    predictions = probabilities > 0.5
    control_predictions = control_probabilities > 0.5

    def reference_average_precision(probabilities, labels, average):
        return float(average_precision_score(labels, probabilities, average=average))

    def evaluate(metric):
        task = get_task("sequence_multilabel_classification")
        task.metrics = {"average_precision": metric}
        rng = np.random.default_rng(710019)
        resampled_scores = []
        for _ in range(20):
            indices = bootstrap_indices(len(labels), None, rng)
            scores = full_scores(
                task,
                predictions[indices],
                labels[indices],
                probabilities[indices],
                ["average_precision"],
                average,
            )
            resampled_scores.append(scores["average_precision"])
        observed_scores = full_scores(
            task, predictions, labels, probabilities, ["average_precision"], average
        )
        p_values = permutation_p_values(
            task,
            observed_trained=observed_scores,
            trained_predictions=predictions,
            trained_labels=labels,
            trained_probabilities=probabilities,
            control_predictions=control_predictions,
            control_labels=labels,
            control_probabilities=control_probabilities,
            metric_names=["average_precision"],
            n_permutations=32,
            rng=rng,
            average=average,
        )
        return ci_stats(np.asarray(resampled_scores), 0.025, 0.975), p_values

    expected_ci, expected_p_values = evaluate(reference_average_precision)
    actual_ci, actual_p_values = evaluate(average_precision)
    assert actual_ci == pytest.approx(expected_ci, rel=1e-12, abs=1e-12)
    assert actual_p_values == expected_p_values


@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_multilabel_mcc_resampling_matches_original(average: str) -> None:
    rng = np.random.default_rng(0)
    labels = (rng.random((60, 12)) < rng.uniform(0.01, 0.6, 12)).astype(float)
    predictions = (rng.random(labels.shape) < 0.3).astype(int)
    control_predictions = (rng.random(labels.shape) < 0.3).astype(int)

    def reference_multilabel_mcc(predictions, labels, average):
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

    def evaluate(metric):
        task = get_task("sequence_multilabel_classification")
        task.metrics = {"multilabel_mcc": metric}
        rng = np.random.default_rng(710019)
        resampled_scores = []
        for _ in range(20):
            indices = bootstrap_indices(len(labels), None, rng)
            scores = full_scores(
                task,
                predictions[indices],
                labels[indices],
                None,
                ["multilabel_mcc"],
                average,
            )
            resampled_scores.append(scores["multilabel_mcc"])
        observed_scores = full_scores(
            task, predictions, labels, None, ["multilabel_mcc"], average
        )
        p_values = permutation_p_values(
            task,
            observed_trained=observed_scores,
            trained_predictions=predictions,
            trained_labels=labels,
            trained_probabilities=None,
            control_predictions=control_predictions,
            control_labels=labels,
            control_probabilities=None,
            metric_names=["multilabel_mcc"],
            n_permutations=32,
            rng=rng,
            average=average,
        )
        return ci_stats(np.asarray(resampled_scores), 0.025, 0.975), p_values

    assert evaluate(multilabel_mcc) == evaluate(reference_multilabel_mcc)


def test_dense_bootstrap_resample_keeps_numpy_arrays() -> None:
    task = MagicMock(is_ragged=False)
    task.compute_metrics.return_value = {"accuracy": 0.5}
    task.flatten_for_metrics.side_effect = lambda values: values
    predictions = np.arange(4)
    labels = np.array([0, 1, 0, 1])
    probabilities = np.ones((4, 2))
    indices = np.array([3, 1, 1, 0])

    scores = full_scores(
        task,
        predictions[indices],
        labels[indices],
        probabilities[indices],
        ["accuracy"],
        "weighted",
    )

    assert scores == {"accuracy": 0.5}
    call_kwargs = task.compute_metrics.call_args.kwargs
    sampled_predictions = call_kwargs["predictions"]
    sampled_labels = call_kwargs["labels"]
    assert isinstance(sampled_predictions, np.ndarray)
    assert isinstance(sampled_labels, np.ndarray)


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
def test_classification_metrics_match_sklearn(average: str) -> None:
    labels = np.array([0, 2, 2, 0, 2, 0])
    predictions = np.array([0, 1, 0, 2, 2, 0])
    task = get_task("sequence_classification")

    scores = task.compute_metrics(
        predictions=predictions,
        labels=labels,
        metric_names=["accuracy", "f1", "mcc"],
        average=average,
    )

    assert scores["accuracy"] == pytest.approx(accuracy_score(labels, predictions))
    assert scores["f1"] == pytest.approx(
        f1_score(labels, predictions, average=average, zero_division=0)
    )
    assert scores["mcc"] == pytest.approx(matthews_corrcoef(labels, predictions))


@pytest.mark.parametrize("n_examples", [3, 4])
def test_multilabel_permutations_swap_whole_example_rows(n_examples):
    task = MagicMock(is_ragged=False)
    task.flatten_for_metrics.side_effect = lambda values: values
    task.compute_metrics.side_effect = lambda **kwargs: {
        "accuracy": float(np.mean(kwargs["predictions"] == kwargs["labels"]))
    }
    trained = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])[:n_examples]
    control = 1 - trained
    labels = trained.copy()
    control_labels = np.zeros_like(labels)
    probabilities = trained.astype(float)
    control_probabilities = control.astype(float)
    metric_names = ["accuracy"]
    observed = full_scores(
        task, trained, labels, probabilities, metric_names, "weighted"
    )
    seed = 0
    swap = np.random.default_rng(seed).random(n_examples) < 0.5
    expected_trained = np.where(swap[:, None], control, trained)
    expected_control = np.where(swap[:, None], trained, control)
    expected_labels = np.where(swap[:, None], control_labels, labels)
    expected_control_labels = np.where(swap[:, None], labels, control_labels)
    observed_control = float(np.mean(control == control_labels))
    permuted_difference = float(np.mean(expected_trained == expected_labels)) - float(
        np.mean(expected_control == expected_control_labels)
    )
    expected_p_value = (
        1 + int(permuted_difference >= observed["accuracy"] - observed_control)
    ) / 2

    result = permutation_p_values(
        task=task,
        observed_trained=observed,
        trained_predictions=trained,
        trained_labels=labels,
        trained_probabilities=probabilities,
        control_predictions=control,
        control_labels=control_labels,
        control_probabilities=control_probabilities,
        metric_names=metric_names,
        n_permutations=1,
        rng=np.random.default_rng(seed),
        average="weighted",
    )

    assert result == {"accuracy": expected_p_value}
    permuted_calls = task.compute_metrics.call_args_list[-2:]
    np.testing.assert_array_equal(
        permuted_calls[0].kwargs["predictions"], expected_trained
    )
    np.testing.assert_array_equal(permuted_calls[0].kwargs["labels"], expected_labels)
    np.testing.assert_array_equal(
        permuted_calls[0].kwargs["probabilities"], expected_trained
    )
    np.testing.assert_array_equal(
        permuted_calls[1].kwargs["predictions"], expected_control
    )
    np.testing.assert_array_equal(
        permuted_calls[1].kwargs["labels"], expected_control_labels
    )
    np.testing.assert_array_equal(
        permuted_calls[1].kwargs["probabilities"], expected_control
    )


def _task_with_failing_metric(failing: str) -> MagicMock:
    """A task whose *failing* metric is undefined on any input."""
    task = MagicMock(is_ragged=False)
    task.flatten_for_metrics.side_effect = lambda values: values

    def compute(predictions, labels, metric_names, average, probabilities=None):
        (name,) = metric_names
        if name == failing:
            raise ValueError(f"{name} is undefined")
        return {name: float(np.mean(np.asarray(predictions) == np.asarray(labels)))}

    task.compute_metrics.side_effect = compute
    return task


def test_undefined_metric_does_not_suppress_other_metrics() -> None:
    task = _task_with_failing_metric("roc_auc")
    metric_names = ["accuracy", "roc_auc"]
    trained = np.array([0, 1, 0, 1, 1, 0])
    labels = np.array([0, 1, 0, 1, 0, 1])
    control = np.ones(6, dtype=int)

    observed = full_scores(task, trained, labels, None, metric_names, "weighted")
    assert observed["accuracy"] == pytest.approx(4 / 6)
    assert math.isnan(observed["roc_auc"])

    p_values = permutation_p_values(
        task,
        observed,
        trained,
        labels,
        None,
        control,
        labels,
        None,
        metric_names,
        20,
        np.random.default_rng(0),
        "weighted",
    )

    assert p_values["roc_auc"] is None
    assert 0.0 < p_values["accuracy"] <= 1.0


def test_bootstrap_resamples_repeated_ids_as_whole_groups() -> None:
    assert id_groups(None) is None
    assert id_groups(["a", "b"]) is None
    groups = id_groups(["a", "a", "b", "b", "b", "c"])
    np.testing.assert_array_equal(groups, [0, 0, 1, 1, 1, 2])
    rng = np.random.default_rng(0)
    for _ in range(20):
        counts = np.bincount(
            bootstrap_indices(len(groups), groups, rng), minlength=len(groups)
        )
        assert counts[0] == counts[1]
        assert counts[2] == counts[3] == counts[4]


def test_two_sided_permutation_is_symmetric() -> None:
    task = _task_with_failing_metric("unused")
    labels = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    better = labels.copy()
    worse = 1 - labels

    def p_value(first, second):
        observed = full_scores(task, first, labels, None, ["accuracy"], "weighted")
        return permutation_p_values(
            task,
            observed,
            first,
            labels,
            None,
            second,
            labels,
            None,
            ["accuracy"],
            200,
            np.random.default_rng(0),
            "weighted",
            two_sided=True,
        )["accuracy"]

    assert p_value(better, worse) == p_value(worse, better) < 0.05


def test_upsert_long_df_replaces_only_matching_variant(tmp_path) -> None:
    path = tmp_path / "scores_long.parquet"

    def rows(variant: str, value: float) -> pl.DataFrame:
        return pl.DataFrame({"variant": [variant], "metric": ["f1"], "value": [value]})

    write_parquet(
        upsert_long_df(path, rows("mlp-frozen_trunk", 0.1), "mlp-frozen_trunk"), path
    )
    write_parquet(
        upsert_long_df(path, rows("knn-frozen_trunk", 0.2), "knn-frozen_trunk"), path
    )
    merged = upsert_long_df(path, rows("mlp-frozen_trunk", 0.9), "mlp-frozen_trunk")

    by_variant = dict(zip(merged["variant"].to_list(), merged["value"].to_list()))
    assert by_variant == {"knn-frozen_trunk": 0.2, "mlp-frozen_trunk": 0.9}


def test_upsert_long_df_retains_other_split_for_same_variant(tmp_path) -> None:
    path = tmp_path / "scores_long.parquet"
    first = pl.DataFrame(
        {"variant": ["mlp-frozen_trunk"], "split": ["identity_30"], "value": [0.7]}
    )
    write_parquet(first, path)
    second = pl.DataFrame(
        {"variant": ["mlp-frozen_trunk"], "split": ["identity_50"], "value": [0.8]}
    )

    combined = upsert_long_df(
        path, second, variant="mlp-frozen_trunk", split="identity_50"
    )

    assert combined.sort("split")["value"].to_list() == [0.7, 0.8]


def test_benjamini_hochberg_is_grouped_per_metric() -> None:
    task = get_task("sequence_classification")
    metric_names = ["accuracy", "f1"]
    rng = np.random.default_rng(3)
    labels = np.array([0, 1] * 12)
    trained = labels.copy()
    trained[:3] = 1 - trained[:3]
    control_arrays = {
        "untrained": (np.zeros(24, dtype=int), labels, None),
        "scrambled_labels": (trained, rng.permutation(labels), None),
    }

    resample_rows = []
    control_rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in control_arrays
    }
    for i in range(8):
        idx = rng.integers(0, 24, size=24)
        row = full_scores(
            task, trained[idx], labels[idx], None, metric_names, "weighted"
        )
        row["resample"] = i
        resample_rows.append(row)
        for name, (c_preds, c_labels, _) in control_arrays.items():
            c_row = full_scores(
                task, c_preds[idx], c_labels[idx], None, metric_names, "weighted"
            )
            c_row["resample"] = i
            control_rows[name].append(c_row)

    comparisons = summarize_controls(
        resample_rows,
        control_rows,
        metric_names,
        0.025,
        0.975,
        task,
        trained,
        labels,
        None,
        control_arrays,
        20,
        np.random.default_rng(11),
        "weighted",
    )

    controls = list(control_arrays)
    for metric in metric_names:
        raw = [comparisons[c][metric]["p_value"] for c in controls]
        adjusted = [comparisons[c][metric]["p_value_adjusted"] for c in controls]
        assert adjusted == pytest.approx(benjamini_hochberg(raw))


def _build_prepared_artifacts(
    model_family: ModelFamily,
    task_type: str,
    num_labels: int = 2,
    batch_size: int = 2,
) -> PreparedArtifacts:
    tokenizer = model_family.tokenizer
    dataset = tokenize_dataset(
        build_raw_dataset_dict(task_type, num_labels=num_labels),
        tokenizer=tokenizer,
        task_type=task_type,
        sequence_column="sequence",
        label_column="targets",
        max_length=64,
        num_proc=1,
        random_truncate=False,
    )
    collator = EvaluationCollator(
        tokenizer=tokenizer,
        task_type=task_type,
        config=EvaluationCollatorConfig(label_column="targets"),
    )
    train_dl, val_dl, test_dl = build_dataloaders(
        dataset=dataset,
        collator=collator,
        config=DataLoaderConfig(
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        ),
    )
    effective_num_labels = 1 if task_type == "sequence_regression" else num_labels
    model = build_tiny_model(model_family, task_type, effective_num_labels)
    return PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
    )


def _make_workspace(tmp_path, task_type, num_labels=2):
    target_type = "continuous" if task_type == "sequence_regression" else "categorical"
    label_format = "per_token" if task_type == "token_classification" else "scalar"
    dataset_num_labels = None if task_type == "sequence_regression" else num_labels
    config = EvaluationWorkspaceConfig(
        run_id="test",
        dataset_name="toy_dataset",
        model_name="toy-model",
        task_type=task_type,
        target_type=target_type,
        label_format=label_format,
        num_labels=dataset_num_labels,
        base_path=str(tmp_path / "eval"),
    )
    workspace = get_evaluation_workspace(config)
    workspace.setup_directories()
    return workspace


# ---------------------------------------------------------------------------
# ScoreConfig validation
# ---------------------------------------------------------------------------


class TestScoreConfig:
    def test_defaults(self):
        cfg = ScoreConfig()
        assert cfg.metrics is None
        assert cfg.metric_aggregation == "weighted"
        assert cfg.bootstrap_enabled is False
        assert cfg.n_samples == 1000
        assert cfg.n_permutations == 1000
        assert cfg.confidence_interval == 0.95

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ScoreConfig(not_a_real_field=1)


# ---------------------------------------------------------------------------
# ScoreStep.run — point-estimate metrics (offline, tiny real models)
# ---------------------------------------------------------------------------


class TestScoreStepMetrics:
    @pytest.mark.parametrize(
        ("task_type", "expected_metrics"),
        [
            ("sequence_classification", {"accuracy", "f1", "mcc", "roc_auc"}),
            (
                "token_classification",
                {"accuracy", "f1", "precision", "recall", "mcc"},
            ),
            ("sequence_regression", {"mse", "mae", "r2", "pearsonr", "spearmanr"}),
        ],
    )
    def test_default_metrics_are_valid_for_task(
        self, tmp_path, model_family, task_type, expected_metrics
    ):
        prepared = _build_prepared_artifacts(model_family, task_type)
        workspace = _make_workspace(tmp_path, task_type)
        PredictStep(
            PredictConfig(
                controls=[],
                collect_probabilities=(task_type == "sequence_classification"),
            )
        ).run(workspace, prepared)

        result = ScoreStep(ScoreConfig()).run(workspace)

        assert set(result.scores) == expected_metrics

    def test_sequence_classification_writes_scores(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(collect_probabilities=True, controls=[])).run(
            workspace, prepared
        )

        config = ScoreConfig(metrics=["accuracy", "f1", "roc_auc"])
        result = ScoreStep(config).run(workspace)

        assert set(result.scores) == {"accuracy", "f1", "roc_auc"}
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())
        assert result.scores_path is not None and result.scores_path.exists()
        assert result.scores_path == workspace.run_manifest_path
        manifest = json.loads(result.scores_path.read_text())
        on_disk = manifest["score"]
        assert on_disk["scores"] == result.scores
        assert on_disk["metadata"]["model_name"] == "toy-model"
        assert on_disk["metadata"]["dataset_name"] == "toy_dataset"
        assert on_disk["metadata"]["task_type"] == "sequence_classification"
        assert on_disk["metadata"]["split"] == "default"
        assert result.metadata is not None
        assert result.metadata.model_name == "toy-model"

    def test_predict_and_score_sections_coexist_in_run_manifest(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        ScoreStep(ScoreConfig(metrics=["accuracy"])).run(workspace)

        manifest = json.loads(workspace.run_manifest_path.read_text())
        assert set(manifest) == {"predict", "score"}
        assert "controls" in manifest["predict"]["config"]
        assert manifest["score"]["scores"]["accuracy"] is not None

    def test_score_can_use_in_memory_predictions_without_disk_manifest(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        predicted = PredictStep(PredictConfig(controls=[], persist_to_disk=False)).run(
            workspace, prepared
        )

        assert predicted.predictions_path is None
        assert not workspace.run_manifest_path.exists()
        result = ScoreStep(ScoreConfig(metrics=["accuracy"])).run(
            workspace, predict_result=predicted
        )

        assert set(result.scores) == {"accuracy"}

    def test_sequence_classification_rejects_precision_recall(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        config = ScoreConfig(metrics=["precision"])
        with pytest.raises(KeyError):
            ScoreStep(config).run(workspace)

    def test_token_classification_flattens_valid_positions(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "token_classification")
        workspace = _make_workspace(tmp_path, "token_classification")
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        result = ScoreStep(ScoreConfig()).run(workspace)

        assert set(result.scores) == {"accuracy", "f1", "precision", "recall", "mcc"}
        assert -1.0 <= result.scores["mcc"] <= 1.0
        assert all(
            0.0 <= result.scores[name] <= 1.0
            for name in ("accuracy", "f1", "precision", "recall")
        )

    def test_sequence_regression_computes_regression_metrics(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(
            model_family, "sequence_regression", num_labels=1
        )
        workspace = _make_workspace(tmp_path, "sequence_regression")
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        config = ScoreConfig(metrics=["mse", "r2", "pearsonr", "spearmanr"])
        result = ScoreStep(config).run(workspace)

        assert set(result.scores) == {"mse", "r2", "pearsonr", "spearmanr"}

    def test_requires_predictions(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        predicted = PredictStep(PredictConfig(controls=[])).run(workspace, prepared)
        assert predicted.predictions_path is not None
        # Simulate PredictStep having produced no predictions, by overwriting
        # the just-written parquet with an empty frame of the same schema.
        write_parquet(
            pl.DataFrame(schema=predicted.predictions.schema),
            predicted.predictions_path,
        )

        with pytest.raises(ValueError, match="predictions"):
            ScoreStep(ScoreConfig()).run(workspace)


# ---------------------------------------------------------------------------
# ScoreStep.run — bootstrap confidence intervals / control comparisons
# ---------------------------------------------------------------------------


class TestScoreStepBootstrap:
    def test_all_degenerate_bootstrap_resamples_do_not_abort(self):
        interval = ci_stats(np.array([np.nan, np.nan]), 0.025, 0.975)

        assert all(math.isnan(interval[name]) for name in ("mean", "lower", "upper"))

    def test_bootstrap_disabled_by_default(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        prediction = PredictStep(PredictConfig(controls=["random_trunk"])).run(
            workspace, prepared
        )

        step = ScoreStep(ScoreConfig(metrics=["accuracy", "f1"]))
        result = step.run(workspace)
        condition_path = workspace.run_root / "condition_scores.parquet"
        conditions = pl.read_parquet(condition_path)
        assert set(conditions["condition"]) == {"trained", "random_trunk"}
        expected = full_scores(
            get_task("sequence_classification", workspace.num_labels),
            prediction.controls["random_trunk"]["prediction"].to_list(),
            prediction.predictions["label"].to_list(),
            None,
            ["accuracy", "f1"],
            "macro",
        )
        actual = conditions.filter(pl.col("condition") == "random_trunk")
        assert dict(zip(actual["metric"], actual["score"])) == expected
        step.run(workspace)
        assert pl.read_parquet(condition_path).equals(conditions)

        assert result.confidence_intervals is None
        assert result.control_comparisons is None

    @pytest.mark.parametrize(
        "control", ["random_trunk", "random_head", "random_both", "scrambled_sequences"]
    )
    def test_model_based_controls_produce_comparisons(
        self, tmp_path, model_family, control
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=[control])).run(workspace, prepared)

        config = ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
        result = ScoreStep(config).run(workspace)

        assert result.control_comparisons is not None
        assert control in result.control_comparisons
        comparison = result.control_comparisons[control]["accuracy"]
        assert {
            "trained_mean",
            "control_mean",
            "diff_mean",
            "diff_lower",
            "diff_upper",
            "p_value",
            "p_value_adjusted",
        } <= set(comparison)
        assert 0.0 <= comparison["p_value"] <= 1.0
        assert 0.0 <= comparison["p_value_adjusted"] <= 1.0

        manifest = json.loads(workspace.run_manifest_path.read_text())
        assert (
            manifest["bootstrap"]["confidence_intervals"] == result.confidence_intervals
        )
        assert (
            manifest["bootstrap"]["control_comparisons"] == result.control_comparisons
        )
        assert "predict" in manifest
        assert "score" in manifest

        # The same comparisons remain available in tidy form.
        assert result.long_path is not None
        on_disk = pl.read_parquet(result.long_path)
        assert control in on_disk["control"].to_list()

    def test_n_permutations_is_independent_of_n_samples(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=["random_trunk"])).run(workspace, prepared)

        config = ScoreConfig(
            metrics=["accuracy"],
            bootstrap_enabled=True,
            n_samples=20,
            n_permutations=3,
        )
        result = ScoreStep(config).run(workspace)

        comparison = result.control_comparisons["random_trunk"]["accuracy"]
        # With only 3 permutations, the p-value can only take multiples of 1/4.
        assert comparison["p_value"] in (0.25, 0.5, 0.75, 1.0)

    def test_same_seed_is_deterministic(self, tmp_path, model_family):
        control = "random_trunk"
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=[control])).run(
            workspace, prepared, seed=123
        )

        config = ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
        first = ScoreStep(config).run(workspace, seed=123)
        second = ScoreStep(config).run(workspace, seed=123)

        assert first.confidence_intervals == second.confidence_intervals
        assert first.control_comparisons == second.control_comparisons

    def test_majority_class_control_produces_comparison(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=["majority_class"])).run(workspace, prepared)

        result = ScoreStep(
            ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
        ).run(workspace)

        assert result.control_comparisons is not None
        assert "majority_class" in result.control_comparisons

    def test_train_mean_control_produces_comparison(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_regression")
        workspace = _make_workspace(tmp_path, "sequence_regression")
        PredictStep(PredictConfig(controls=["train_mean"])).run(workspace, prepared)

        result = ScoreStep(
            ScoreConfig(metrics=["mse"], bootstrap_enabled=True, n_samples=5)
        ).run(workspace)

        assert result.control_comparisons is not None
        assert "train_mean" in result.control_comparisons

    def test_no_controls_yields_no_comparisons(self, tmp_path, model_family):
        # When no controls are requested at predict time, ScoreStep still
        # produces point-estimate metrics and confidence intervals, but no
        # control comparisons.
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")

        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        config = ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=3)
        result = ScoreStep(config).run(workspace)

        assert result.control_comparisons is None
        assert set(result.confidence_intervals) == {"accuracy"}
        manifest = json.loads(workspace.run_manifest_path.read_text())
        assert manifest["bootstrap"]["control_comparisons"] == {}


# ---------------------------------------------------------------------------
# ScoreStep.run — tidy long-format output (for cross-model plotting)
# ---------------------------------------------------------------------------


class TestScoreStepLongDf:
    def test_written_without_bootstrap(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        result = ScoreStep(ScoreConfig(metrics=["accuracy", "f1"])).run(workspace)

        assert result.long_path is not None and result.long_path.exists()
        assert result.long_df is not None
        on_disk = pl.read_parquet(result.long_path)
        assert on_disk.equals(result.long_df)
        assert set(on_disk["metric"]) == {"accuracy", "f1"}
        assert on_disk["control"].is_null().all()
        assert on_disk["model_name"].unique().to_list() == ["toy-model"]
        assert on_disk["dataset_name"].unique().to_list() == ["toy_dataset"]
        assert set(on_disk.filter(pl.col("metric") == "accuracy")["value"]) == {
            result.scores["accuracy"]
        }
        # No bootstrap: CI columns should be null.
        assert on_disk["ci_mean"].is_null().all()

    def test_includes_confidence_intervals_and_control_comparisons(
        self, tmp_path, model_family
    ):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=["random_head"])).run(workspace, prepared)

        config = ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
        result = ScoreStep(config).run(workspace)

        long_df = result.long_df
        assert long_df is not None

        base_row = long_df.filter(pl.col("control").is_null()).row(0, named=True)
        assert base_row["ci_lower"] <= base_row["ci_mean"] <= base_row["ci_upper"]

        control_row = long_df.filter(pl.col("control") == "random_head").row(
            0, named=True
        )
        expected = result.control_comparisons["random_head"]["accuracy"]
        assert control_row["diff_mean"] == expected["diff_mean"]
        assert control_row["p_value"] == expected["p_value"]

    def test_build_long_df_handles_missing_optional_fields(self):
        df = build_long_df(
            metadata=None,
            scores={"accuracy": 0.5},
            confidence_intervals=None,
            control_comparisons=None,
        )
        assert df.to_dicts() == [
            {
                "model_name": None,
                "model_repo_id": None,
                "dataset_name": None,
                "dataset_repo_id": None,
                "task_type": None,
                "split": None,
                "head_type": None,
                "layer_index": None,
                "variant": None,
                "seed": None,
                "metric": "accuracy",
                "control": None,
                "value": 0.5,
                "ci_mean": None,
                "ci_lower": None,
                "ci_upper": None,
                "control_mean": None,
                "diff_mean": None,
                "diff_lower": None,
                "diff_upper": None,
                "p_value": None,
                "p_value_adjusted": None,
            }
        ]


# ---------------------------------------------------------------------------
# ScoreStep.run — across all task types (not just sequence_classification)
# ---------------------------------------------------------------------------


TASK_TYPE_METRICS = [
    ("sequence_classification", ["accuracy", "f1"]),
    ("token_classification", ["accuracy", "f1"]),
    ("sequence_regression", ["mse", "r2"]),
]


class TestScoreStepAcrossTaskTypes:
    @pytest.mark.parametrize(("task_type", "metrics"), TASK_TYPE_METRICS)
    def test_confidence_intervals_bracket_point_estimate(
        self, tmp_path, model_family, task_type, metrics
    ):
        prepared = _build_prepared_artifacts(model_family, task_type)
        workspace = _make_workspace(tmp_path, task_type)
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        config = ScoreConfig(metrics=metrics, bootstrap_enabled=True, n_samples=10)
        result = ScoreStep(config).run(workspace)

        assert set(result.confidence_intervals) == set(metrics)
        for name in metrics:
            ci = result.confidence_intervals[name]
            assert ci["lower"] <= ci["mean"] <= ci["upper"]
        assert result.output_path is not None
        assert result.output_path.exists()

    @pytest.mark.parametrize(("task_type", "metrics"), TASK_TYPE_METRICS)
    def test_scrambled_labels_control(self, tmp_path, model_family, task_type, metrics):
        prepared = _build_prepared_artifacts(model_family, task_type)
        workspace = _make_workspace(tmp_path, task_type)
        PredictStep(PredictConfig(controls=["scrambled_labels"])).run(
            workspace, prepared
        )

        config = ScoreConfig(metrics=metrics, bootstrap_enabled=True, n_samples=5)
        result = ScoreStep(config).run(workspace)

        assert result.control_comparisons is not None
        assert set(result.control_comparisons["scrambled_labels"]) == set(metrics)
        # No model inference needed for this control: nothing should be cached.
        assert not (
            workspace.preds_path / "control_scrambled_labels_predictions.parquet"
        ).exists()


# ---------------------------------------------------------------------------
# utils.bootstrap.benjamini_hochberg — post-hoc multiple-comparisons correction
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_adjusted_values_are_at_least_raw_values(self):
        raw = [0.01, 0.2, 0.03, 0.5, 0.001]
        adjusted = benjamini_hochberg(raw)

        assert len(adjusted) == len(raw)
        for raw_p, adjusted_p in zip(raw, adjusted):
            assert adjusted_p >= raw_p - 1e-12
            assert 0.0 <= adjusted_p <= 1.0

    def test_matches_known_bh_result(self):
        # Textbook example: 5 raw p-values -> BH q-values computed by hand.
        # q_(i) = min_{j>=i} (p_(j) * n / j), enforced non-decreasing.
        raw = [0.01, 0.04, 0.03, 0.005, 0.5]
        n = len(raw)
        sorted_raw = sorted(raw)
        expected_sorted = []
        running_min = 1.0
        for rank in range(n, 0, -1):
            p = sorted_raw[rank - 1]
            running_min = min(running_min, p * n / rank)
            expected_sorted.append(running_min)
        expected_sorted.reverse()

        adjusted = benjamini_hochberg(raw)
        order = sorted(range(n), key=lambda i: raw[i])
        adjusted_sorted_by_raw = [adjusted[i] for i in order]

        for expected, actual in zip(expected_sorted, adjusted_sorted_by_raw):
            assert actual == pytest.approx(expected)

    def test_monotonic_in_sorted_order(self):
        raw = [0.2, 0.001, 0.15, 0.049, 0.049, 0.9, 0.0001]
        adjusted = benjamini_hochberg(raw)
        order = sorted(range(len(raw)), key=lambda i: raw[i])
        adjusted_sorted = [adjusted[i] for i in order]

        assert adjusted_sorted == sorted(adjusted_sorted)

    def test_empty_input(self):
        assert benjamini_hochberg([]) == []


@pytest.mark.parametrize(
    "task_name",
    [
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    ],
)
@pytest.mark.parametrize("average", ["macro", "weighted"])
def test_classification_scoring_preserves_list_metrics(task_name, average):
    task = get_task(task_name)
    rng = np.random.default_rng(17)
    probabilities = None
    if task_name == "sequence_multilabel_classification":
        labels = rng.integers(0, 2, size=(12, 4)).tolist()
        probabilities = rng.random((12, 4)).tolist()
        predictions = (np.asarray(probabilities) > 0.5).astype(int).tolist()
    elif task_name == "token_classification":
        labels = [[0, 2], [], [1, 2, 0], [2]]
        predictions = [[0, 1], [], [2, 2, 0], [0]]
    else:
        labels = [0, 2, 1, 0, 1, 2]
        probabilities = rng.random((len(labels), 3))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1).tolist()
        probabilities = probabilities.tolist()

    metric_names = task.default_metrics()
    expected = task.compute_metrics(
        task.flatten_for_metrics(predictions),
        task.flatten_for_metrics(labels),
        metric_names,
        average,
        probabilities,
    )
    actual = full_scores(
        task, predictions, labels, probabilities, metric_names, average
    )

    assert actual == expected


@pytest.mark.parametrize(
    "task_name",
    [
        "sequence_classification",
        "sequence_multilabel_classification",
        "token_classification",
    ],
)
def test_shared_control_labels_preserve_bootstrap_results(
    tmp_path, monkeypatch, task_name
):
    from modules.evaluate.src.steps import score as score_module

    if task_name == "sequence_multilabel_classification":
        labels = [[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 1]]
        predictions = [[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 1]]
        metrics = ["f1", "multilabel_mcc"]
        label_format = "multilabel"
    elif task_name == "token_classification":
        labels = [[0, 2], [], [1, 2, 0], [2]]
        predictions = [[0, 1], [], [2, 2, 0], [0]]
        metrics = ["accuracy", "f1", "mcc"]
        label_format = "per_token"
    else:
        labels = [0, 1, 2, 0, 1, 2, 0, 1]
        predictions = [0, 2, 2, 1, 1, 0, 0, 1]
        metrics = ["accuracy", "f1", "mcc"]
        label_format = "scalar"
    controls = (
        [row[::-1] for row in predictions]
        if task_name == "token_classification"
        else predictions[::-1]
    )
    ids = list(range(len(labels)))
    predict_output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name="toy-model", dataset_name="toy_dataset", task_type=task_name
        ),
        predictions=pl.DataFrame(
            {"id": ids, "prediction": predictions, "label": labels}
        ),
        controls={"untrained": pl.DataFrame({"id": ids, "prediction": controls})},
        requested_controls=["untrained", "scrambled_labels"],
    )
    config = ScoreConfig(
        metrics=metrics, bootstrap_enabled=True, n_samples=16, n_permutations=16
    )

    def run(name):
        workspace = get_evaluation_workspace(
            EvaluationWorkspaceConfig(
                run_id="test",
                model_name="toy-model",
                dataset_name="toy_dataset",
                task_type=task_name,
                target_type="categorical",
                label_format=label_format,
                num_labels=3,
                base_path=str(tmp_path / name),
            )
        )
        workspace.setup_directories()
        return ScoreStep(config).run(workspace, seed=123, predict_result=predict_output)

    collect = score_module.collect_control_arrays

    def copy_labels(*args, **kwargs):
        return {
            name: (predictions, list(labels), probabilities)
            for name, (predictions, labels, probabilities) in collect(
                *args, **kwargs
            ).items()
        }

    # Independent label objects exercise the original copying path.
    with monkeypatch.context() as reference:
        reference.setattr(score_module, "collect_control_arrays", copy_labels)
        expected = run("independent")
    actual = run("shared")

    np.testing.assert_equal(actual.scores, expected.scores)
    assert actual.resamples.equals(expected.resamples)
    np.testing.assert_equal(actual.confidence_intervals, expected.confidence_intervals)
    np.testing.assert_equal(actual.control_comparisons, expected.control_comparisons)
    assert actual.long_df.equals(expected.long_df)
