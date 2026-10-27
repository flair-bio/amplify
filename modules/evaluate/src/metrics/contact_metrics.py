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

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from modules.evaluate.src.tasks.contact_prediction import (
        ContactAlignment,
        ContactUnit,
    )

CASP_METRICS = ("AUC", "P@L", "P@L/2", "P@L/5", "ROC-AUC")
CASP_SEPARATIONS = (6, 24)


def contact_prediction_row(
    scores: np.ndarray | None, unit: ContactUnit, alignment: ContactAlignment
) -> dict[str, Any]:
    """Keep only eligible predictions, with original coordinates for later scoring."""
    from modules.evaluate.src.tasks.contact_prediction import candidate_pairs

    pairs = candidate_pairs(unit, alignment)
    if not unit.labels_available:
        status = "missing_labels"
    elif not pairs.length:
        status = "no_retained_members"
    elif not len(pairs.i):
        status = "no_eligible_pairs"
    else:
        status = "scorable"
    values = np.empty(0, dtype=np.float64)
    if status == "scorable":
        if scores is None:
            raise ValueError("Available candidates require full-chain predictions")
        scores = np.asarray(scores)
        length = len(alignment.retained_positions)
        if scores.shape != (length, length):
            raise ValueError(
                "Predictions must cover the full retained chain without cropping"
            )
        values = np.asarray(scores[pairs.i, pairs.j], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Eligible contact pairs require finite predictions")
    return {
        "length": pairs.length,
        "status": status,
        "original_i": pairs.original_i.tolist(),
        "original_j": pairs.original_j.tolist(),
        "labels": pairs.labels.tolist(),
        "scores": values.tolist(),
    }


def score_contact_prediction(row: dict[str, Any], minsep: int = 6) -> dict[str, Any]:
    """Rank one self-contained prediction row at an original-coordinate separation."""
    if minsep not in CASP_SEPARATIONS:
        raise ValueError("CASP minsep must be 6 or 24")
    original_i = np.asarray(row["original_i"], dtype=np.int64)
    original_j = np.asarray(row["original_j"], dtype=np.int64)
    truth = np.asarray(row["labels"])
    values = np.asarray(row["scores"], dtype=np.float64)
    if any(array.ndim != 1 for array in (original_i, original_j, truth, values)):
        raise ValueError("Contact prediction columns must be one-dimensional")
    if not (len(original_i) == len(original_j) == len(truth) == len(values)):
        raise ValueError("Contact prediction columns have different lengths")
    if (
        np.any(original_i < 0)
        or np.any(original_j <= original_i)
        or not np.isin(truth, [0, 1]).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Contact predictions require ordered indices, binary labels and finite scores"
        )
    length = row["length"]
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 0:
        raise ValueError("Contact length must be a nonnegative integer")
    length = int(length)
    keep = original_j - original_i >= minsep
    original_i, original_j = original_i[keep], original_j[keep]
    truth, values = truth[keep], values[keep]
    result: dict[str, Any] = dict.fromkeys(CASP_METRICS)
    result.update(length=length, n_pairs=len(truth), n_positives=int(truth.sum()))
    status = row["status"]
    if status not in {
        "scorable",
        "missing_labels",
        "no_retained_members",
        "no_eligible_pairs",
    }:
        raise ValueError(f"Unknown contact status: {status}")
    if status != "scorable":
        if len(truth):
            raise ValueError("Unscorable contact rows must have no candidate pairs")
        return dict(result, status=status)
    if not length:
        raise ValueError("Scorable contact rows require retained members")
    if not len(truth):
        return dict(result, status="no_eligible_pairs")
    order = np.lexsort((original_j, original_i, -values))
    cumulative = truth[order].cumsum()
    cutoffs = np.minimum(np.maximum(np.arange(1, 11) * length // 10, 1), len(truth))
    precision = cumulative[cutoffs - 1] / cutoffs
    result.update(
        {
            "AUC": float(precision.mean()),
            "P@L": float(precision[9]),
            "P@L/2": float(precision[4]),
            "P@L/5": float(precision[1]),
            "status": "scored",
        }
    )
    if len(np.unique(truth)) == 2:
        result["ROC-AUC"] = float(roc_auc_score(truth, values))
    return result


def score_contact_unit(
    scores: np.ndarray | None,
    unit: ContactUnit,
    alignment: ContactAlignment,
    minsep: int = 6,
) -> dict[str, Any]:
    return score_contact_prediction(
        contact_prediction_row(scores, unit, alignment), minsep
    )


def _mean_contact_metrics(
    records: Iterable[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, int]]:
    records = list(records)
    means, counts = {}, {}
    for metric in CASP_METRICS:
        values = [row[metric] for row in records if row[metric] is not None]
        if not np.isfinite(values).all():
            raise ValueError("Undefined contact metrics must be None, not NaN/infinity")
        means[metric] = float(np.mean(values)) if values else None
        counts[metric] = len(values)
    return means, counts


def aggregate_contact_scores(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equal-weight EUs, chains and complexes; never pool pairs or fitting seeds."""
    records = list(records)
    identity_fields = (
        "dataset_fingerprint",
        "schema_version",
        "protocol_version",
        "model_name",
        "model_repo_id",
        "task_type",
        "variant",
        "seed",
    )
    for field in identity_fields:
        if len({row.get(field) for row in records}) > 1:
            raise ValueError(f"Cannot aggregate mixed {field}")
    identity = {
        field: records[0][field]
        for field in identity_fields
        if records and field in records[0]
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    seen = set()
    chain_metadata: dict[tuple[str, str], tuple[int | None, str | None]] = {}
    for row in records:
        if (
            row["kind"] not in ("global", "intra_eu")
            or row["minsep"] not in CASP_SEPARATIONS
        ):
            raise ValueError("Unknown CASP setting or separation")
        group = (row["split"], row["edition"], row["kind"], row["minsep"])
        key = (*group, row["record_id"], row["unit_id"])
        if key in seen:
            raise ValueError(f"Duplicate scoring unit: {key}")
        seen.add(key)
        chain_key = (row["split"], row["record_id"])
        metadata = (row["edition"], row["complex_id"])
        if chain_key in chain_metadata and chain_metadata[chain_key] != metadata:
            raise ValueError("Inconsistent edition/complex metadata within a chain")
        chain_metadata[chain_key] = metadata
        groups[group].append(row)
    summaries = []
    for (split, edition, kind, minsep), units in sorted(groups.items()):
        sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in units:
            sequences[row["record_id"]].append(row)
        complexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        sequence_means, sequence_counts = [], []
        for rows in sequences.values():
            if kind == "global" and len(rows) != 1:
                raise ValueError("Global scoring requires one unit per chain")
            means, counts = _mean_contact_metrics(rows)
            sequence_means.append(means)
            sequence_counts.append(counts)
            if edition is not None:
                if not rows[0]["complex_id"]:
                    raise ValueError("CASP aggregation requires complex_id")
                complexes[rows[0]["complex_id"]].append(means)
        complex_means = [_mean_contact_metrics(rows)[0] for rows in complexes.values()]
        means, complex_counts = _mean_contact_metrics(complex_means)
        if edition is None:
            means = _mean_contact_metrics(sequence_means)[0]
        summaries.append(
            dict(
                identity,
                split=split,
                edition=edition,
                kind=kind,
                minsep=minsep,
                **means,
                n_units=len(units),
                n_sequences=len(sequences),
                n_complexes=len(complexes),
                contributing_units=_mean_contact_metrics(units)[1],
                contributing_sequences={
                    metric: sum(count[metric] > 0 for count in sequence_counts)
                    for metric in CASP_METRICS
                },
                contributing_complexes=complex_counts,
            )
        )
    return summaries


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
