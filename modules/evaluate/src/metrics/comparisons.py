"""Pure statistical helpers behind ScoreStep's bootstrap confidence intervals
and trained-vs-control comparisons: array conversion, per-resample/full-set
metric scoring, the paired permutation test, and multiple-comparisons
correction. Kept out of score.py since none of these need ScoreStep's own
state (config/workspace/IO) -- each takes everything it needs as a plain
argument, so they're independently testable and reusable.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import false_discovery_control

from modules.evaluate.src.contracts.artifacts import (
    ControlComparison,
)
from modules.evaluate.src.metrics.bootstrap import (
    confusion_scores,
    full_scores,
    mean_per_example_scores,
)
from modules.evaluate.src.metrics.metrics import LOWER_IS_BETTER_METRICS
from modules.evaluate.src.tasks.base import FlatMetricsTask

logger = logging.getLogger(__name__)


def permutation_p_values(
    task: FlatMetricsTask,
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
        confusion_scores(
            control_confusions.sum(axis=0),
            metric_names,
            average,
            include_absent=task.class_labels is not None,
        )
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
                include_absent=task.class_labels is not None,
            )
            perm_control = confusion_scores(
                np.where(swap_matrices, trained_confusions, control_confusions).sum(
                    axis=0
                ),
                metric_names,
                average,
                include_absent=task.class_labels is not None,
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


def summarize_controls(
    resample_rows: list[dict[str, float]],
    control_resample_rows: dict[str, list[dict[str, float]]],
    metric_names: list[str],
    lower_q: float,
    upper_q: float,
    task: FlatMetricsTask,
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
        confusion_scores(
            trained_confusions.sum(axis=0),
            metric_names,
            average,
            include_absent=task.class_labels is not None,
        )
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
