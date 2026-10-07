"""
Contact-prediction metrics: separation-binned precision-at-L and a pooled
contact AUC.

Unlike the pointwise metrics in ``metrics.py``, these operate on the
per-sequence *ragged* structure directly rather than a flattened
scalar sequence (see ``tasks.contact_prediction.ContactPredictionTask``,
whose ``flatten_for_metrics`` is an identity for this reason). Each row is
``{"scores": ..., "separations": ...}`` / ``{"length": ..., "values": ...}``,
where ``length`` is the sequence length ``L`` needed for the top-``L/k``
cutoff and the lists are aligned by position.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np

# Standard CASP/contact-prediction separation bins, in tokenized-residue
# positions.
SEPARATION_BINS: dict[str, tuple[int, int | None]] = {
    "short": (6, 12),
    "medium": (12, 24),
    "long": (24, None),
}
# Top-k cutoffs as a fraction of sequence length L: "l" -> L, "l2" -> L/2, etc.
TOPK_FRACTIONS: dict[str, int] = {"l": 1, "l2": 2, "l5": 5}
# Deciles of L used by the legacy per-range "AUC" (see contact_range_auc).
_DECILES: tuple[float, ...] = tuple(d / 10 for d in range(1, 11))


def _iter_rows(
    predictions: Any, labels: Any
) -> Iterator[tuple[int, list[float], list[int], list[int]]]:
    """Yield aligned sequence length, scores, separations, and labels per row."""
    for pred_row, label_row in zip(predictions, labels):
        if not isinstance(pred_row, dict):
            yield (
                int(label_row[0]),
                [float(score) for score, _ in pred_row[1:]],
                [int(sep) for _, sep in pred_row[1:]],
                [int(value) for value in label_row[1:]],
            )
            continue
        yield (
            int(label_row["length"]),
            [float(score) for score in pred_row["scores"]],
            [int(sep) for sep in pred_row["separations"]],
            [int(value) for value in label_row["values"]],
        )


def precision_at_l(
    predictions: Any,
    labels: Any,
    average: str,
    *,
    min_sep: int,
    max_sep: int | None,
    fraction: int,
) -> float:
    """Mean, over sequences, of precision among the top ``L/fraction``
    highest-scored candidate pairs whose separation falls in
    ``[min_sep, max_sep)``."""
    precisions = []
    for length, scores, separations, values in _iter_rows(predictions, labels):
        candidates = [
            (score, label)
            for score, sep, label in zip(scores, separations, values)
            if sep >= min_sep and (max_sep is None or sep < max_sep)
        ]
        if not candidates or length <= 0:
            continue
        k = max(1, round(length / fraction))
        top = sorted(candidates, key=lambda item: item[0], reverse=True)[:k]
        precisions.append(sum(label for _, label in top) / len(top))
    return float(np.mean(precisions)) if precisions else float("nan")


def contact_range_auc(
    predictions: Any,
    labels: Any,
    average: str,
    *,
    min_sep: int,
    max_sep: int | None,
) -> float:
    """Legacy SequenceStructureEvaluator "AUC": not a real ROC-AUC, but the
    mean, over the 10 deciles of L, of precision among the top
    ``round(decile * L)`` scored candidates in ``[min_sep, max_sep)``
    (matches ``apperitif.utils.metrics.compute_precisions``'s
    ``binned_precisions.mean(-1)``, averaged per-sequence like
    ``precision_at_l``)."""
    per_sequence_means = []
    for length, scores, separations, values in _iter_rows(predictions, labels):
        candidates = [
            (score, label)
            for score, sep, label in zip(scores, separations, values)
            if sep >= min_sep and (max_sep is None or sep < max_sep)
        ]
        if not candidates or length <= 0:
            continue
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        decile_precisions = []
        for decile in _DECILES:
            k = max(1, round(length * decile))
            top = ranked[:k]
            decile_precisions.append(sum(label for _, label in top) / len(top))
        per_sequence_means.append(float(np.mean(decile_precisions)))
    return float(np.mean(per_sequence_means)) if per_sequence_means else float("nan")


def _make_precision_metric(
    bin_name: str, fraction_name: str
) -> Callable[[Any, Any, str], float]:
    min_sep, max_sep = SEPARATION_BINS[bin_name]
    fraction = TOPK_FRACTIONS[fraction_name]

    def metric(predictions: Any, labels: Any, average: str) -> float:
        return precision_at_l(
            predictions,
            labels,
            average,
            min_sep=min_sep,
            max_sep=max_sep,
            fraction=fraction,
        )

    metric.__name__ = f"contact_precision_at_{fraction_name}_{bin_name}"
    return metric


# e.g. "contact_precision_at_l_long", "contact_precision_at_l5_short", ...
PRECISION_METRICS = {
    f"contact_precision_at_{fraction_name}_{bin_name}": _make_precision_metric(
        bin_name, fraction_name
    )
    for bin_name in SEPARATION_BINS
    for fraction_name in TOPK_FRACTIONS
}


def _make_range_auc_metric(bin_name: str) -> Callable[[Any, Any, str], float]:
    min_sep, max_sep = SEPARATION_BINS[bin_name]

    def metric(predictions: Any, labels: Any, average: str) -> float:
        return contact_range_auc(
            predictions, labels, average, min_sep=min_sep, max_sep=max_sep
        )

    metric.__name__ = f"contact_auc_{bin_name}"
    return metric


# e.g. "contact_auc_long", "contact_auc_short", ...
RANGE_AUC_METRICS = {
    f"contact_auc_{bin_name}": _make_range_auc_metric(bin_name)
    for bin_name in SEPARATION_BINS
}
