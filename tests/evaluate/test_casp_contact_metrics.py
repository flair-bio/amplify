"""CASP ranking, original separation and hierarchical averaging."""

from itertools import combinations

import numpy as np
import pytest

from modules.evaluate.src.metrics.contact_metrics import (
    CASP_METRICS,
    aggregate_contact_scores,
    score_contact_unit,
)
from modules.evaluate.src.tasks.contact_prediction import prepare_contact_dataset

from .test_contact_data import hf_data, row


def chain(**overrides):
    return prepare_contact_dataset(hf_data([row(**overrides)])).splits["test"][0]


def test_floor_cutoffs_original_index_ties_and_distinct_auc():
    record = chain(sequence="A" * 11, positive_pairs=[[1, 7]])
    result = score_contact_unit(
        np.zeros((11, 11), dtype=np.uint8), record.units[0], record.alignment
    )
    assert result["n_pairs"] == 15
    assert result["P@L/2"] == 0  # Floor(11/2)=5; the positive ranks sixth.
    assert result["P@L/5"] == 0
    assert result["P@L"] == pytest.approx(1 / 11)
    assert result["AUC"] == pytest.approx(sum(1 / k for k in (6, 7, 8, 9, 11)) / 10)
    assert result["ROC-AUC"] == 0.5


def test_available_pair_denominators_zero_positives_and_unscorable():
    record = chain(
        sequence="A" * 11, valid_positions=[0, 6, 7, 8], positive_pairs=[[0, 8]]
    )
    result = score_contact_unit(np.zeros((11, 11)), record.units[0], record.alignment)
    assert result["n_pairs"] == 3 and result["length"] == 11
    assert result["P@L"] == pytest.approx(1 / 3)
    assert result["AUC"] == pytest.approx(8 / 30)
    for positive_pairs, expected in ([], 0), ([[0, 6], [0, 7], [0, 8]], 1):
        zero = chain(
            sequence="A" * 11,
            valid_positions=[0, 6, 7, 8],
            positive_pairs=positive_pairs,
        )
        scored = score_contact_unit(np.zeros((11, 11)), zero.units[0], zero.alignment)
        assert scored["AUC"] == expected and scored["ROC-AUC"] is None
    unscored = score_contact_unit(
        np.zeros((11, 11)), record.units[0], record.alignment, 24
    )
    assert unscored["status"] == "no_eligible_pairs"
    assert all(unscored[metric] is None for metric in CASP_METRICS)


def test_scoring_matches_independent_original_coordinate_oracle():
    sequence = "AXAAAAXAAA" * 4
    valid = list(range(0, len(sequence), 2))
    positives = [[i, j] for i, j in combinations(valid, 2) if (i + j) % 3 == 0]
    global_row = row(sequence=sequence, valid_positions=valid, positive_pairs=positives)
    members = [*range(11), *range(22, 38)]
    eu = row(
        sequence=sequence,
        kind="intra_eu",
        unit_id="D1",
        positions=members,
        valid_positions=[i for i in valid if i in members],
        positive_pairs=[[i, j] for i, j in positives if i in members and j in members],
    )
    record = prepare_contact_dataset(hf_data([global_row, eu])).splits["test"][0]
    positions = record.alignment.retained_positions
    scores = np.random.default_rng(42).integers(0, 4, (len(positions), len(positions)))
    for source, unit in zip((global_row, eu), record.units):
        for minsep in (6, 24):
            eligible = [
                (float(scores[a, b]), i, j, [i, j] in source["positive_pairs"])
                for a, i in enumerate(positions)
                for b, j in enumerate(positions)
                if j - i >= minsep
                and i in source["valid_positions"]
                and j in source["valid_positions"]
            ]
            ranked = sorted(eligible, key=lambda item: (-item[0], item[1], item[2]))
            length = sum(
                source["positions"] is None or i in source["positions"]
                for i in positions
            )
            precision = []
            for decile in range(1, 11):
                k = min(max(1, decile * length // 10), len(ranked))
                precision.append(sum(item[3] for item in ranked[:k]) / k)
            actual = score_contact_unit(scores, unit, record.alignment, minsep)
            assert actual["length"] == length and actual["n_pairs"] == len(ranked)
            assert actual["AUC"] == pytest.approx(np.mean(precision))
            assert actual["P@L"] == precision[-1]
            assert actual["P@L/2"] == precision[4]
            assert actual["P@L/5"] == precision[1]


def test_eligible_predictions_must_be_finite():
    record = chain()
    scores = np.zeros((31, 31))
    scores[0, 0] = np.nan
    assert (
        score_contact_unit(scores, record.units[0], record.alignment)["status"]
        == "scored"
    )
    scores[0, 6] = np.inf
    with pytest.raises(ValueError, match="finite"):
        score_contact_unit(scores, record.units[0], record.alignment)


def score_row(record, unit, value, *, complex_id="C1", edition=14, kind="intra_eu"):
    return dict(
        split="test",
        edition=edition,
        kind=kind,
        minsep=6,
        record_id=record,
        unit_id=unit,
        complex_id=complex_id,
        **dict.fromkeys(CASP_METRICS, value),
    )


def test_hierarchy_averages_units_chains_and_complexes_separately():
    rows = [
        score_row("A", "D1", 1.0),
        score_row("A", "D2", 0.0),
        score_row("A", "missing", None),
        score_row("B", "D1", 0.0),
        score_row("C", "D1", 1.0, complex_id="C2"),
        score_row("unknown", "D1", None, complex_id="C3"),
        score_row("A15", "D1", 0.0, edition=15),
    ]
    rows[0]["ROC-AUC"] = None
    result = aggregate_contact_scores(rows)
    assert len(result) == 2
    edition14 = result[0]
    assert edition14["AUC"] == 0.625  # ((0.5+0)/2 + 1)/2.
    assert edition14["ROC-AUC"] == 0.5
    assert edition14["n_units"] == 6 and edition14["n_sequences"] == 4
    assert edition14["n_complexes"] == 3
    assert edition14["contributing_units"]["AUC"] == 4
    assert edition14["contributing_units"]["ROC-AUC"] == 3
    assert edition14["contributing_sequences"]["AUC"] == 3
    assert edition14["contributing_complexes"]["AUC"] == 2
    assert result[1]["AUC"] == 0
    global_rows = [
        score_row("A", "chain", 1.0, kind="global"),
        score_row("B", "chain", 0.0, kind="global"),
        score_row("C", "chain", 1.0, kind="global", complex_id="C2"),
    ]
    assert aggregate_contact_scores(global_rows)[0]["AUC"] == 0.75
    development = [
        dict(r, split="validation", edition=None, complex_id=None) for r in global_rows
    ]
    assert aggregate_contact_scores(development)[0]["AUC"] == pytest.approx(2 / 3)


def test_aggregation_rejects_duplicate_units_and_mixed_run_identities():
    record = score_row("A", "D1", 1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_contact_scores([record, record])
    for field in ("seed", "task_type", "dataset_fingerprint"):
        with pytest.raises(ValueError, match=f"mixed {field}"):
            aggregate_contact_scores(
                [dict(record, **{field: 1}), dict(record, **{field: 2})]
            )
    with pytest.raises(ValueError, match="one unit"):
        aggregate_contact_scores(
            [dict(record, kind="global"), dict(record, kind="global", unit_id="other")]
        )
