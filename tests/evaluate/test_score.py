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
    f1_score,
    matthews_corrcoef,
)

from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import DataLoaderConfig, build_dataloaders
from modules.evaluate.src.dataset.dataloader import tokenize_dataset
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.steps.predict import PredictConfig, PredictStep
from modules.evaluate.src.steps.score import ScoreConfig, ScoreStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.bootstrap import (
    benjamini_hochberg,
    ci_stats,
    full_scores,
    permutation_p_values,
    summarize_controls,
)
from modules.evaluate.src.utils.long_format import build_long_df, upsert_long_df
from modules.evaluate.src.utils.io import write_parquet

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model


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
        PredictStep(PredictConfig(
            controls=[],
            collect_probabilities=(task_type == "sequence_classification"),
        )).run(workspace, prepared)

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
        predicted = (
            PredictStep(PredictConfig(controls=[], persist_to_disk=False)).run(
                workspace, prepared
            )
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
        PredictStep(PredictConfig(controls=[])).run(workspace, prepared)

        result = ScoreStep(ScoreConfig(metrics=["accuracy", "f1"])).run(workspace)

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

        # Control comparisons are only persisted in tidy form, via long_df.
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
        PredictStep(PredictConfig(controls=[control])).run(workspace, prepared, seed=123)

        config = ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
        first = ScoreStep(config).run(workspace, seed=123)
        second = ScoreStep(config).run(workspace, seed=123)

        assert first.confidence_intervals == second.confidence_intervals
        assert first.control_comparisons == second.control_comparisons

    def test_majority_class_control_produces_comparison(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_classification")
        workspace = _make_workspace(tmp_path, "sequence_classification")
        PredictStep(PredictConfig(controls=["majority_class"])).run(workspace, prepared)

        result = (
            ScoreStep(
                ScoreConfig(metrics=["accuracy"], bootstrap_enabled=True, n_samples=5)
            ).run(workspace)
        )

        assert result.control_comparisons is not None
        assert "majority_class" in result.control_comparisons

    def test_train_mean_control_produces_comparison(self, tmp_path, model_family):
        prepared = _build_prepared_artifacts(model_family, "sequence_regression")
        workspace = _make_workspace(tmp_path, "sequence_regression")
        PredictStep(PredictConfig(controls=["train_mean"])).run(workspace, prepared)

        result = (
            ScoreStep(
                ScoreConfig(metrics=["mse"], bootstrap_enabled=True, n_samples=5)
            ).run(workspace)
        )

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
        PredictStep(PredictConfig(controls=["scrambled_labels"])).run(workspace, prepared)

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
