"""Tests for the contact_prediction task: label alignment/collation,
per-sequence row extraction, metrics, and the bilinear contact model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.evaluate.src.dataset.contact import (
    ContactChain,
    align_contact_sequence,
    separation_prior_scores,
)
from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.model.inference import pad_for_gather
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.token_classification import IGNORE_INDEX


def _task():
    return get_task("contact_prediction")


def test_separation_prior_uses_original_coordinates_across_unresolved_residues():
    chain = ContactChain(
        record_id="target",
        split="test",
        source="casp",
        edition=14,
        complex_id="complex",
        target="target",
        alignment=align_contact_sequence("ACXDE"),
        units=(),
    )
    scores = separation_prior_scores(chain)
    assert scores.shape == (4, 4)
    assert scores[0, 1] == pytest.approx(1 / 2)
    assert scores[0, 2] == pytest.approx(1 / 4)
    assert scores[0, 1] > scores[0, 2]


def test_pad_for_gather_pads_both_contact_map_axes():
    class TwoProcessAccelerator:
        num_processes = 2

        def __init__(self):
            self.dims: list[int] = []

        def pad_across_processes(self, tensor, *, dim, pad_index):
            self.dims.append(dim)
            return tensor

    accelerator = TwoProcessAccelerator()
    contact_map = torch.zeros((2, 8, 8))

    result = pad_for_gather(contact_map, accelerator, IGNORE_INDEX, dims=(1, 2))

    assert result is contact_map
    assert accelerator.dims == [1, 2]


def test_precision_at_l_ranks_within_bin_and_uses_length():
    # length=10 -> top-L/5 = top-2 candidates in the "long" bin (sep>=24).
    predictions = [{"scores": [0.9, 0.1, 0.8], "separations": [30, 25, 26]}]
    labels = [{"length": 10, "values": [1, 0, 0]}]
    score = contact_metrics.precision_at_l(
        predictions, labels, "macro", min_sep=24, max_sep=None, fraction=5
    )
    # top-2 by score: 0.9 (label 1), 0.8 (label 0) -> precision 0.5
    assert score == 0.5


def test_contact_range_auc_averages_decile_precisions_per_sequence():
    # length=10, sep>=24 ("long" bin): 4 candidates ranked by score
    # (0.9->1, 0.8->0, 0.7->1, 0.1->0). Golden value for the mean, over the
    # 10 deciles of L, of precision among the top round(decile * L) of them.
    predictions = [{"scores": [0.9, 0.1, 0.8, 0.7], "separations": [30, 25, 26, 28]}]
    labels = [{"length": 10, "values": [1, 0, 1, 0]}]
    auc = contact_metrics.contact_range_auc(
        predictions, labels, "macro", min_sep=24, max_sep=None
    )
    assert auc == pytest.approx(0.6166666666666666)


@pytest.mark.parametrize("legacy_rows", [False, True])
@pytest.mark.parametrize("bin_name", list(contact_metrics.SEPARATION_BINS))
@pytest.mark.parametrize("fraction", [1, 2, 5, None])
def test_contact_ranking_matches_python_reference(legacy_rows, bin_name, fraction):
    rng = np.random.default_rng(48)
    predictions = [
        {
            "scores": rng.integers(0, 5, size=size).astype(float).tolist(),
            "separations": rng.integers(1, 40, size=size).tolist(),
        }
        for size in [0, 1, 7, 31, 100]
    ]
    labels = [
        {"length": size // 2, "values": rng.integers(0, 2, size=size).tolist()}
        for size in [0, 1, 7, 31, 100]
    ]
    predictions += [
        {"scores": [0.5] * 6, "separations": [5, 6, 11, 12, 23, 24]},
        {
            "scores": [float("nan"), 0.2, 0.9, float("nan"), float("inf")],
            "separations": [24] * 5,
        },
        {"scores": [0.9, 0.1, 0.8], "separations": [6, 12]},
        {"scores": ["0.9", "0.2"], "separations": ["6", "24"]},
    ]
    labels += [
        {"length": 5, "values": [0, 1, 0, 1, 0, 1]},
        {"length": 7, "values": [1, 0, 1, 0, 1]},
        {"length": 3, "values": [1]},
        {"length": 5, "values": ["1", "0"]},
    ]
    min_sep, max_sep = contact_metrics.SEPARATION_BINS[bin_name]
    per_sequence = []
    for prediction, label in zip(predictions, labels):
        candidates = [
            (score, value)
            for score, sep, value in zip(
                map(float, prediction["scores"]),
                map(int, prediction["separations"]),
                map(int, label["values"]),
            )
            if sep >= min_sep and (max_sep is None or sep < max_sep)
        ]
        length = label["length"]
        if not candidates or length <= 0:
            continue
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        cutoffs = (
            [max(1, round(length / fraction))]
            if fraction is not None
            else [max(1, round(length * (decile / 10))) for decile in range(1, 11)]
        )
        precisions = [
            sum(value for _, value in ranked[:k]) / len(ranked[:k]) for k in cutoffs
        ]
        per_sequence.append(
            precisions[0] if fraction is not None else float(np.mean(precisions))
        )
    expected = float(np.mean(per_sequence)) if per_sequence else float("nan")
    if legacy_rows:
        predictions = [
            [0] + list(zip(row["scores"], row["separations"])) for row in predictions
        ]
        labels = [[row["length"]] + row["values"] for row in labels]
    kwargs = {"min_sep": min_sep, "max_sep": max_sep}
    if fraction is None:
        actual = contact_metrics.contact_range_auc(
            predictions, labels, "macro", **kwargs
        )
    else:
        actual = contact_metrics.precision_at_l(
            predictions, labels, "macro", fraction=fraction, **kwargs
        )
    np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "value, error",
    [("invalid", ValueError), (float("nan"), ValueError), (None, TypeError)],
)
def test_contact_metrics_reject_invalid_labels(value, error):
    predictions = [{"scores": [0.5], "separations": [24]}]
    labels = [{"length": 5, "values": [value]}]
    with pytest.raises(error):
        contact_metrics.precision_at_l(
            predictions, labels, "macro", min_sep=24, max_sep=None, fraction=1
        )
    with pytest.raises(error):
        contact_metrics.contact_range_auc(
            predictions, labels, "macro", min_sep=24, max_sep=None
        )
