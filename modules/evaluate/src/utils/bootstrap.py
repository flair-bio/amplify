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
from scipy.stats import false_discovery_control

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.schemas.artifacts import (
    ConfidenceInterval,
    ControlComparison,
    PredictOutput,
)
from modules.evaluate.src.tasks import TaskHandler
from modules.evaluate.src.utils.hpo import LOWER_IS_BETTER_METRICS

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
    task: TaskHandler,
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
        if (
            not len(flat_preds)
            or flat_preds.dtype.kind not in "iu"
            or flat_labels.dtype.kind not in "iu"
        ):
            return None
        flattened[name] = (flat_preds, flat_labels, pred_lengths)

    classes = np.unique(
        np.concatenate(
            [
                array
                for preds, labels, _ in flattened.values()
                for array in (preds, labels)
            ]
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
    matrix: np.ndarray, metric_names: list[str], average: str
) -> dict[str, float]:
    """Pooled classification metrics, with sklearn's observed-class/zero rules."""
    actual = matrix.sum(axis=1, dtype=float)
    predicted = matrix.sum(axis=0, dtype=float)
    total = actual.sum()
    if total == 0:
        return dict.fromkeys(metric_names, float("nan"))
    present = (actual + predicted) != 0
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
) -> dict[str, dict[str, float]]:
    """Draw the same protein multiplicities as bootstrap_indices, without rows."""
    n_groups = len(next(iter(matrices.values())))
    counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
    return {
        condition: confusion_scores(
            np.einsum("g,gij->ij", counts, values), metric_names, average
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
    task: TaskHandler,
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


def permutation_p_values(
    task: TaskHandler,
    observed_trained: dict[str, float],
    trained_predictions: np.ndarray,
    trained_labels: np.ndarray,
    trained_probabilities: np.ndarray | None,
    control_predictions: np.ndarray,
    control_labels: np.ndarray,
    control_probabilities: np.ndarray | None,
    metric_names: list[str],
    n_permutations: int,
    rng: np.random.Generator,
    average: str,
    groups: np.ndarray | None = None,
    two_sided: bool = False,
    *,
    trained_per_example_scores: np.ndarray | None = None,
    control_per_example_scores: np.ndarray | None = None,
    trained_confusions: np.ndarray | None = None,
    control_confusions: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Paired randomization-test p-values for "the trained model did not
    beat this control", one per metric.

    Under the null hypothesis that the two conditions are equivalent, each
    example's ``(prediction, label[, probability])`` tuple is exchangeable
    between the trained model and the control. Every permutation
    independently swaps that tuple between the two conditions with
    probability 0.5, recomputes each condition's metric over the full test
    set, and takes the difference; the p-value is the
    continuity-corrected (``(k + 1) / (n + 1)``) fraction of permutations
    whose difference is at least as favorable to "trained beats control"
    as the observed difference. Unlike a bootstrap tail probability, this
    samples an actual null distribution, so the result is a valid
    frequentist p-value that Benjamini-Hochberg can legitimately correct.

    ``observed_trained`` is precomputed once by the caller and shared
    across every control (it doesn't depend on which control this call is
    scoring against). Metrics are accounted for independently: a metric
    that is undefined (NaN) on the full set or on a given permutation is
    skipped for that permutation only, and returns ``None`` if no
    permutation ever yielded a usable value.

    ``groups`` (from :func:`id_groups`) swaps rows sharing a source ID
    together. ``two_sided`` tests for any difference, using ``|diff|``.
    Optional per-example scores must be supplied for both conditions, in
    metric_names column order; swapping these rows avoids metric recomputation.
    Optional confusion matrices must likewise be supplied for both conditions,
    with the same protein/class axes; swaps then operate on pooled counts.
    """
    if (trained_per_example_scores is None) != (control_per_example_scores is None):
        raise ValueError("Both conditions must supply per-example scores.")
    if (trained_confusions is None) != (control_confusions is None):
        raise ValueError("Both conditions must supply confusion matrices.")

    observed_control = (
        confusion_scores(control_confusions.sum(axis=0), metric_names, average)
        if control_confusions is not None
        else full_scores(
            task,
            control_predictions,
            control_labels,
            control_probabilities,
            metric_names,
            average,
            per_example_scores=control_per_example_scores,
        )
    )

    observed_diff = {
        name: observed_trained[name] - observed_control[name] for name in metric_names
    }
    at_least_as_extreme = {name: 0 for name in metric_names}
    valid = {name: 0 for name in metric_names}

    n_examples = len(trained_predictions)
    has_probs = trained_probabilities is not None and control_probabilities is not None
    for _ in range(n_permutations):
        group_swap = (
            rng.random(n_examples if groups is None else int(groups.max()) + 1) < 0.5
        )
        if trained_confusions is not None and control_confusions is not None:
            swap_matrices = group_swap[:, None, None]
            perm_trained = confusion_scores(
                np.where(swap_matrices, control_confusions, trained_confusions).sum(
                    axis=0
                ),
                metric_names,
                average,
            )
            perm_control = confusion_scores(
                np.where(swap_matrices, trained_confusions, control_confusions).sum(
                    axis=0
                ),
                metric_names,
                average,
            )
        elif (
            trained_per_example_scores is not None
            and control_per_example_scores is not None
        ):
            swap = group_swap if groups is None else group_swap[groups]
            swap_rows = swap[:, None]
            perm_trained = mean_per_example_scores(
                np.where(
                    swap_rows, control_per_example_scores, trained_per_example_scores
                ),
                metric_names,
            )
            perm_control = mean_per_example_scores(
                np.where(
                    swap_rows, trained_per_example_scores, control_per_example_scores
                ),
                metric_names,
            )
        else:
            swap = group_swap if groups is None else group_swap[groups]
            prediction_swap = swap[:, None] if trained_predictions.ndim == 2 else swap
            label_swap = swap[:, None] if trained_labels.ndim == 2 else swap
            perm_trained_preds = np.where(
                prediction_swap, control_predictions, trained_predictions
            )
            perm_control_preds = np.where(
                prediction_swap, trained_predictions, control_predictions
            )
            # Shared labels are unchanged by a paired swap.
            if trained_labels is control_labels:
                perm_trained_labels = perm_control_labels = trained_labels
            else:
                perm_trained_labels = np.where(
                    label_swap, control_labels, trained_labels
                )
                perm_control_labels = np.where(
                    label_swap, trained_labels, control_labels
                )
            if has_probs:
                swap_2d = swap[:, None]
                perm_trained_probs = np.where(
                    swap_2d, control_probabilities, trained_probabilities
                )
                perm_control_probs = np.where(
                    swap_2d, trained_probabilities, control_probabilities
                )
            else:
                perm_trained_probs = perm_control_probs = None

            perm_trained = full_scores(
                task,
                perm_trained_preds,
                perm_trained_labels,
                perm_trained_probs,
                metric_names,
                average,
            )
            perm_control = full_scores(
                task,
                perm_control_preds,
                perm_control_labels,
                perm_control_probs,
                metric_names,
                average,
            )

        for name in metric_names:
            diff = perm_trained[name] - perm_control[name]
            # A NaN diff compares False against everything, which would
            # otherwise inflate significance rather than be ignored.
            if np.isnan(diff) or np.isnan(observed_diff[name]):
                continue
            valid[name] += 1
            if two_sided:
                if abs(diff) >= abs(observed_diff[name]):
                    at_least_as_extreme[name] += 1
            elif name in LOWER_IS_BETTER_METRICS:
                if diff <= observed_diff[name]:
                    at_least_as_extreme[name] += 1
            else:
                if diff >= observed_diff[name]:
                    at_least_as_extreme[name] += 1

    return {
        name: (
            (at_least_as_extreme[name] + 1) / (valid[name] + 1) if valid[name] else None
        )
        for name in metric_names
    }


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


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Compute Benjamini-Hochberg FDR-adjusted p-values (q-values) for a
    list of raw p-values, as a post-hoc correction for the multiple
    (control, metric) hypothesis tests run by a single score call.

    Thin wrapper around ``scipy.stats.false_discovery_control`` (BH is
    its default method), returning adjusted values in the same order as
    `p_values`.
    """
    adjusted = false_discovery_control(np.asarray(p_values, dtype=float))
    return adjusted.tolist()


def collect_control_arrays(
    predict_result: PredictOutput,
    predictions_col: list[Any],
    labels_col: list[Any],
    probabilities_col: list[Any] | None,
    rng: np.random.Generator,
    task: TaskHandler,
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


def summarize_controls(
    resample_rows: list[dict[str, float]],
    control_resample_rows: dict[str, list[dict[str, float]]],
    metric_names: list[str],
    lower_q: float,
    upper_q: float,
    task: TaskHandler,
    predictions_arr: np.ndarray,
    labels_arr: np.ndarray,
    probabilities_arr: np.ndarray | None,
    control_arrays_np: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    n_permutations: int,
    rng: np.random.Generator,
    average: str,
    groups: np.ndarray | None = None,
    *,
    trained_per_example_scores: np.ndarray | None = None,
    control_per_example_scores: dict[str, np.ndarray | None] | None = None,
    trained_confusions: np.ndarray | None = None,
    control_confusions: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, ControlComparison]]:
    """Per metric, summarize how the trained model compares to each control.

    ``diff_mean`` and its interval come from the paired bootstrap resamples
    (a valid bootstrap confidence interval for the difference). ``p_value``
    is a separate paired-permutation-test p-value (see
    :func:`permutation_p_values`) for the null hypothesis that the trained
    model and the control are equivalent -- a proper null distribution,
    rather than a bootstrap tail probability. The raw p-values are then
    corrected for multiple comparisons (Benjamini-Hochberg FDR) within each
    metric, across controls: the same metric on the same predictions is one
    family, whereas accuracy/f1/mcc are near-perfectly correlated with each
    other and do not form a valid joint family. Optional per-example scores
    reuse prepared contributions for point estimates and permutations.
    Token confusion matrices similarly reuse grouped counts."""
    trained_by_resample = {row["resample"]: row for row in resample_rows}

    # Independent of `control`, so compute once rather than once per
    # control inside permutation_p_values.
    observed_trained = (
        confusion_scores(trained_confusions.sum(axis=0), metric_names, average)
        if trained_confusions is not None
        else full_scores(
            task,
            predictions_arr,
            labels_arr,
            probabilities_arr,
            metric_names,
            average,
            per_example_scores=trained_per_example_scores,
        )
    )

    control_comparisons: dict[str, dict[str, ControlComparison]] = {}
    # metric -> [(control, raw p-value)], so BH can be applied per metric below.
    p_values_by_metric: dict[str, list[tuple[str, float]]] = {}
    for control, rows in control_resample_rows.items():
        c_preds, c_labels, c_probs = control_arrays_np[control]
        permutation_p = permutation_p_values(
            task,
            observed_trained,
            predictions_arr,
            labels_arr,
            probabilities_arr,
            c_preds,
            c_labels,
            c_probs,
            metric_names,
            n_permutations,
            rng,
            average,
            groups=groups,
            trained_per_example_scores=trained_per_example_scores,
            control_per_example_scores=(
                control_per_example_scores[control]
                if control_per_example_scores is not None
                else None
            ),
            trained_confusions=trained_confusions,
            control_confusions=control_confusions[control]
            if control_confusions is not None
            else None,
        )

        comparisons: dict[str, ControlComparison] = {}
        for name in metric_names:
            paired = [
                (trained_by_resample[row["resample"]][name], row[name])
                for row in rows
                if row["resample"] in trained_by_resample
                and not np.isnan(trained_by_resample[row["resample"]][name])
                and not np.isnan(row[name])
            ]
            p_value = permutation_p.get(name)
            if not paired or p_value is None:
                continue
            trained_vals = np.array([p[0] for p in paired])
            control_vals = np.array([p[1] for p in paired])
            diffs = trained_vals - control_vals
            comparisons[name] = {
                "trained_mean": float(np.mean(trained_vals)),
                "control_mean": float(np.mean(control_vals)),
                "diff_mean": float(np.mean(diffs)),
                "diff_lower": float(np.quantile(diffs, lower_q)),
                "diff_upper": float(np.quantile(diffs, upper_q)),
                "p_value": p_value,
                # Placeholder; filled in below once every comparison's
                # raw p-value has been collected.
                "p_value_adjusted": p_value,
            }
            p_values_by_metric.setdefault(name, []).append((control, p_value))
        if comparisons:
            control_comparisons[control] = comparisons

    for name, entries in p_values_by_metric.items():
        adjusted_p_values = benjamini_hochberg([p for _, p in entries])
        for (control, _), adjusted in zip(entries, adjusted_p_values):
            control_comparisons[control][name]["p_value_adjusted"] = adjusted

    if control_comparisons:
        logger.info("Bootstrap control comparisons: %s", control_comparisons)

    return control_comparisons
