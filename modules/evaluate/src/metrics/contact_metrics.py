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
# Deciles of L used by the binned precision summary.
_DECILES: tuple[float, ...] = tuple(d / 10 for d in range(1, 11))


def _iter_rows(
    predictions: Any, labels: Any
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield aligned sequence length, scores, separations, and labels per row."""
    for pred_row, label_row in zip(predictions, labels):
        if not isinstance(pred_row, dict):
            yield (
                int(label_row[0]),
                np.asarray([score for score, _ in pred_row[1:]], dtype=float),
                np.asarray([sep for _, sep in pred_row[1:]], dtype=int),
                np.asarray(label_row[1:], dtype=int),
            )
            continue
        yield (
            int(label_row["length"]),
            np.asarray(pred_row["scores"], dtype=float),
            np.asarray(pred_row["separations"], dtype=int),
            np.asarray(label_row["values"], dtype=int),
        )


def _ranked_values(
    scores: np.ndarray,
    separations: np.ndarray,
    values: np.ndarray,
    min_sep: int,
    max_sep: int | None,
) -> np.ndarray:
    """Rank labels within a separation bin, retaining input order for ties."""
    size = min(len(scores), len(separations), len(values))
    scores, separations, values = scores[:size], separations[:size], values[:size]
    mask = separations >= min_sep
    if max_sep is not None:
        mask &= separations < max_sep
    scores, values = scores[mask], values[mask]
    if np.isnan(scores).any():
        # Python's ordering of NaNs depends on the original row order.
        return np.asarray(
            [
                value
                for _, value in sorted(
                    zip(scores, values), key=lambda item: item[0], reverse=True
                )
            ],
            dtype=int,
        )
    return values[np.argsort(-scores, kind="stable")]


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
        ranked = _ranked_values(scores, separations, values, min_sep, max_sep)
        if not len(ranked) or length <= 0:
            continue
        k = max(1, round(length / fraction))
        precisions.append(float(ranked[:k].mean()))
    return float(np.mean(precisions)) if precisions else float("nan")


def contact_range_auc(
    predictions: Any,
    labels: Any,
    average: str,
    *,
    min_sep: int,
    max_sep: int | None,
) -> float:
    """Mean precision over ten sequence-length deciles in one separation bin.

    For each sequence, rank candidate pairs in ``[min_sep, max_sep)`` and
    compute precision among the top ``round(decile * L)`` for each decile.
    This metric is a binned precision summary, not ROC-AUC.
    """
    per_sequence_means = []
    for length, scores, separations, values in _iter_rows(predictions, labels):
        ranked = _ranked_values(scores, separations, values, min_sep, max_sep)
        if not len(ranked) or length <= 0:
            continue
        decile_precisions = []
        for decile in _DECILES:
            k = max(1, round(length * decile))
            decile_precisions.append(float(ranked[:k].mean()))
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
