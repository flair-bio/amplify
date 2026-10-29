"""Pure flat-task scoring and paired statistical analysis.

Tasks choose the scientific protocol; the score step publishes returned results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from modules.evaluate.src.contracts.artifacts import (
    ConfidenceInterval,
    PredictOutput,
    ScoreOutput,
)
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.metrics.bootstrap import (
    as_indexable_array,
    bootstrap_confusion_scores,
    bootstrap_indices,
    ci_stats,
    collect_control_arrays,
    confusion_scores,
    full_scores,
    id_groups,
    token_confusion_matrices,
)
from modules.evaluate.src.metrics.comparisons import (
    benjamini_hochberg,
    permutation_p_values,
    summarize_controls,
)
from modules.evaluate.src.metrics.tables import build_long_df

if TYPE_CHECKING:
    from modules.evaluate.src.tasks.base import FlatMetricsTask

logger = logging.getLogger(__name__)


def score_flat(
    task: FlatMetricsTask,
    predict_result: PredictOutput,
    config: ScoreConfig,
    *,
    seed: int,
) -> ScoreOutput:
    """Score aligned rows and, when requested, resample their task-defined units."""
    if predict_result.predictions.is_empty():
        raise ValueError(
            "ScoreStep requires PredictOutput.predictions to be "
            "populated (run the Predict step before scoring)."
        )
    predictions_df = predict_result.predictions
    predictions_col = predictions_df["prediction"].to_list()
    labels_col = predictions_df["label"].to_list()
    probabilities_col = (
        predictions_df["probability"].to_list()
        if "probability" in predictions_df.columns
        else None
    )

    metric_names = config.metrics or task.default_metrics()
    logger.info("Computing metrics: %s", metric_names)
    trained_per_example_scores = task.compute_per_example_metrics(
        predictions_col,
        labels_col,
        metric_names,
        config.metric_aggregation,
        probabilities_col,
    )
    scores = full_scores(
        task,
        predictions_col,
        labels_col,
        probabilities_col,
        metric_names,
        config.metric_aggregation,
        per_example_scores=trained_per_example_scores,
    )
    undefined = sorted(name for name, value in scores.items() if np.isnan(value))
    if undefined:
        logger.warning(
            "Metric(s) %s are undefined on this test split (e.g. a class "
            "absent from the labels) and are recorded as NaN.",
            undefined,
        )
    logger.info("Scores: %s", scores)

    rng = np.random.default_rng(seed)
    condition_scores = {"trained": scores}
    control_arrays = collect_control_arrays(
        predict_result,
        predictions_col,
        labels_col,
        probabilities_col,
        rng,
        task,
    )
    for name, (predictions, labels, probabilities) in control_arrays.items():
        condition_scores[name] = full_scores(
            task,
            predictions,
            labels,
            probabilities,
            metric_names,
            config.metric_aggregation,
        )
    result = ScoreOutput(
        scores=scores,
        metadata=predict_result.metadata,
        config=config.model_dump(mode="json"),
        condition_scores=condition_scores,
    )
    if config.bootstrap_enabled:
        _bootstrap(
            task,
            config,
            predict_result,
            predictions_col,
            labels_col,
            probabilities_col,
            metric_names,
            result,
            rng,
            control_arrays,
            trained_per_example_scores=trained_per_example_scores,
        )
    result.long_df = build_long_df(
        result.metadata,
        result.scores,
        result.confidence_intervals,
        result.control_comparisons,
    )
    return result


def _bootstrap(
    task: FlatMetricsTask,
    config: ScoreConfig,
    predict_result: PredictOutput,
    predictions_col: list[Any],
    labels_col: list[Any],
    probabilities_col: list[Any] | None,
    metric_names: list[str],
    result: ScoreOutput,
    rng: np.random.Generator,
    control_arrays: dict[str, tuple[list[Any], list[Any], list[Any] | None]],
    *,
    trained_per_example_scores: np.ndarray | None,
) -> None:
    n_examples = len(predictions_col)
    predictions_df = predict_result.predictions
    groups = id_groups(
        [task.bootstrap_group_id(i) for i in predictions_df["id"].to_list()]
        if "id" in predictions_df.columns
        else None
    )

    # Pairwise metrics are means of per-protein scores. Prepare each
    # control once; bootstrap samples and permutations then only average.
    control_per_example_scores = {
        control: task.compute_per_example_metrics(
            c_preds,
            c_labels,
            metric_names,
            config.metric_aggregation,
            c_probs,
        )
        if trained_per_example_scores is not None
        else None
        for control, (c_preds, c_labels, c_probs) in control_arrays.items()
    }

    # Convert predictions/labels/probabilities to 1D NumPy arrays once, up
    # front, rather than re-deriving Python lists (via `[col[j] for j in
    # idx]`) on every one of `n_samples` resamples below: array fancy
    # indexing (`arr[idx]`) is done in C and avoids that per-resample,
    # per-element Python-level list allocation.
    ragged = task.is_ragged
    predictions_arr = as_indexable_array(predictions_col, ragged)
    labels_arr = as_indexable_array(labels_col, ragged)
    probabilities_arr = (
        as_indexable_array(probabilities_col, ragged=False)
        if probabilities_col is not None
        else None
    )
    control_arrays_np = {
        control: (
            as_indexable_array(c_preds, ragged),
            (
                labels_arr
                if c_labels is labels_col
                else as_indexable_array(c_labels, ragged)
            ),
            as_indexable_array(c_probs, ragged=False) if c_probs is not None else None,
        )
        for control, (c_preds, c_labels, c_probs) in control_arrays.items()
    }
    confusions = token_confusion_matrices(
        task,
        {
            "trained": (predictions_arr, labels_arr, probabilities_arr),
            **control_arrays_np,
        },
        groups,
        metric_names,
        config.metric_aggregation,
    )
    if confusions is not None:
        logger.info(
            "Using per-protein confusion matrices for token bootstrap and permutations."
        )

    resample_rows: list[dict[str, float]] = []
    control_resample_rows: dict[str, list[dict[str, float]]] = {
        control: [] for control in control_arrays_np
    }

    n_degenerate: dict[str, int] = {name: 0 for name in metric_names}
    for i in range(config.n_samples):
        if confusions is not None:
            scores_by_condition = bootstrap_confusion_scores(
                confusions,
                metric_names,
                config.metric_aggregation,
                rng,
                include_absent=task.class_labels is not None,
            )
            trained_scores = scores_by_condition.pop("trained")
            control_scores_by_name = scores_by_condition
        else:
            idx = bootstrap_indices(n_examples, groups, rng)
            sampled_labels = labels_arr[idx]
            trained_scores = full_scores(
                task,
                predictions_arr[idx],
                sampled_labels,
                probabilities_arr[idx] if probabilities_arr is not None else None,
                metric_names,
                config.metric_aggregation,
                per_example_scores=(
                    trained_per_example_scores[idx]
                    if trained_per_example_scores is not None
                    else None
                ),
            )
            control_scores_by_name = {}
            for control, (c_preds, c_labels, c_probs) in control_arrays_np.items():
                per_example_scores = control_per_example_scores[control]
                control_scores_by_name[control] = full_scores(
                    task,
                    c_preds[idx],
                    sampled_labels if c_labels is labels_arr else c_labels[idx],
                    c_probs[idx] if c_probs is not None else None,
                    metric_names,
                    config.metric_aggregation,
                    per_example_scores=(
                        per_example_scores[idx]
                        if per_example_scores is not None
                        else None
                    ),
                )
        for name in metric_names:
            if np.isnan(trained_scores[name]):
                n_degenerate[name] += 1

        trained_scores["resample"] = i
        resample_rows.append(trained_scores)
        for control, control_scores in control_scores_by_name.items():
            control_scores["resample"] = i
            control_resample_rows[control].append(control_scores)

    for name, count in n_degenerate.items():
        if count:
            logger.warning(
                "Metric '%s' was degenerate (e.g. a class missing from "
                "the resample) in %d/%d bootstrap resamples; those "
                "resamples are excluded from '%s''s confidence interval "
                "only, other metrics are unaffected.",
                name,
                count,
                config.n_samples,
                name,
            )

    resamples_df = pl.DataFrame(resample_rows)

    alpha = 1.0 - config.confidence_interval
    lower_q, upper_q = alpha / 2, 1.0 - alpha / 2

    confidence_intervals: dict[str, ConfidenceInterval] = {
        name: ci_stats(resamples_df[name].to_numpy(), lower_q, upper_q)
        for name in metric_names
    }
    logger.info("Bootstrap confidence intervals: %s", confidence_intervals)

    control_comparisons = summarize_controls(
        resample_rows,
        control_resample_rows,
        metric_names,
        lower_q,
        upper_q,
        task,
        predictions_arr,
        labels_arr,
        probabilities_arr,
        control_arrays_np,
        config.n_permutations,
        rng,
        config.metric_aggregation,
        groups=groups,
        trained_per_example_scores=trained_per_example_scores,
        control_per_example_scores=control_per_example_scores,
        trained_confusions=confusions["trained"] if confusions is not None else None,
        control_confusions=(
            {name: confusions[name] for name in control_arrays_np}
            if confusions is not None
            else None
        ),
    )

    result.confidence_intervals = confidence_intervals
    result.resamples = resamples_df
    result.control_comparisons = control_comparisons or None


def compare_layer_scores(
    task: FlatMetricsTask,
    config: ScoreConfig,
    layer_predictions: dict[int, PredictOutput],
    *,
    variant: str | None,
    seed: int,
) -> pl.DataFrame | None:
    """Compare layers with the deepest reference using paired resampling."""
    if len(layer_predictions) < 2:
        return None
    metric_names = config.metrics or task.default_metrics()
    average = config.metric_aggregation
    reference = max(layer_predictions)
    reference_ids = layer_predictions[reference].predictions["id"].to_list()
    arrays: dict[int, tuple[Any, Any, Any | None]] = {}
    for layer, output in layer_predictions.items():
        frame = output.predictions
        if frame["id"].to_list() != reference_ids:
            raise ValueError(
                f"Layer {layer} predictions are not row-aligned with layer {reference}."
            )
        arrays[layer] = (
            as_indexable_array(frame["prediction"].to_list(), task.is_ragged),
            as_indexable_array(frame["label"].to_list(), task.is_ragged),
            as_indexable_array(frame["probability"].to_list(), ragged=False)
            if "probability" in frame.columns
            else None,
        )
    others = sorted(layer for layer in arrays if layer != reference)
    rng = np.random.default_rng(seed)
    n_examples = len(reference_ids)
    groups = id_groups([task.bootstrap_group_id(i) for i in reference_ids])
    confusions = token_confusion_matrices(
        task,
        {str(layer): values for layer, values in arrays.items()},
        groups,
        metric_names,
        average,
    )
    differences: dict[int, dict[str, list[float]]] = {
        layer: {name: [] for name in metric_names} for layer in others
    }
    for _ in range(config.n_samples):
        if confusions is not None:
            sampled = bootstrap_confusion_scores(
                confusions,
                metric_names,
                average,
                rng,
                include_absent=task.class_labels is not None,
            )
            resampled = {layer: sampled[str(layer)] for layer in arrays}
        else:
            idx = bootstrap_indices(n_examples, groups, rng)
            resampled = {
                layer: full_scores(
                    task,
                    predictions[idx],
                    labels[idx],
                    None if probabilities is None else probabilities[idx],
                    metric_names,
                    average,
                )
                for layer, (predictions, labels, probabilities) in arrays.items()
            }
        for layer in others:
            for name in metric_names:
                differences[layer][name].append(
                    resampled[layer][name] - resampled[reference][name]
                )
    observed = {
        layer: (
            confusion_scores(
                confusions[str(layer)].sum(axis=0),
                metric_names,
                average,
                include_absent=task.class_labels is not None,
            )
            if confusions is not None
            else full_scores(task, *layer_arrays, metric_names, average)
        )
        for layer, layer_arrays in arrays.items()
    }
    alpha = 1.0 - config.confidence_interval
    rows: list[dict[str, Any]] = []
    for layer in others:
        p_values = permutation_p_values(
            task,
            observed[layer],
            *arrays[layer],
            *arrays[reference],
            metric_names,
            config.n_permutations,
            rng,
            average,
            groups=groups,
            two_sided=True,
            trained_confusions=confusions[str(layer)]
            if confusions is not None
            else None,
            control_confusions=confusions[str(reference)]
            if confusions is not None
            else None,
        )
        for name in metric_names:
            interval = ci_stats(
                np.asarray(differences[layer][name]), alpha / 2, 1 - alpha / 2
            )
            rows.append(
                {
                    "variant": variant,
                    "split": layer_predictions[reference].metadata.split,
                    "layer_index": layer,
                    "reference_layer": reference,
                    "metric": name,
                    "score": observed[layer][name],
                    "reference_score": observed[reference][name],
                    "diff_mean": interval["mean"],
                    "diff_lower": interval["lower"],
                    "diff_upper": interval["upper"],
                    "p_value": p_values[name],
                    "p_value_adjusted": None,
                }
            )
    for name in metric_names:
        tested = [
            row for row in rows if row["metric"] == name and row["p_value"] is not None
        ]
        if tested:
            adjusted = benjamini_hochberg([row["p_value"] for row in tested])
            for row, value in zip(tested, adjusted):
                row["p_value_adjusted"] = value
    comparison = pl.DataFrame(rows)
    logger.info("Cross-layer comparison vs layer %d:\n%s", reference, comparison)
    return comparison
