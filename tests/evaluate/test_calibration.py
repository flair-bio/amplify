"""Calibration task: windowing, temperature fitting, scoring, and sourcing."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from datasets import Dataset, DatasetDict
from scipy.special import softmax

from modules.evaluate.scripts.dataset_sourcing.source_calibration import (
    CONSERVATION_EDGES,
    assign_folds,
    build_calibration_frame,
    read_export,
)
from modules.evaluate.src.contracts.artifacts import EvaluationMetadata, PredictOutput
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.dataset.dataloader import tokenize_dataset
from modules.evaluate.src.metrics.bootstrap import id_groups
from modules.evaluate.src.metrics.calibration_metrics import (
    crossfit_temperatures,
    fit_temperature,
    forward_kl,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.calibration import WINDOW_RESIDUES
from modules.evaluate.src.utils.amino_acids import (
    CANONICAL_AMINO_ACIDS,
    resolve_aa_token_ids,
)
from tests.evaluate.conftest import LONG_SEQUENCE

SEQUENCE = (LONG_SEQUENCE * 20)[:1200]
POSITIONS = [5, 600, 1190]
SITE = [1, 0, 0, 1, 0, 1]  # is_t, bin, fold, in_balanced, in_high_nt, mask


def _site(position: int, consensus: str) -> list[float]:
    freqs = [0.0] * 20
    freqs[CANONICAL_AMINO_ACIDS.index(consensus)] = 1.0
    return [float(position), *freqs, *SITE]


def _tokenize(
    tokenizer, max_length: int, random_truncate: bool = False, positions=POSITIONS
):
    raw = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "id": ["P1"],
                    "sequence": [SEQUENCE],
                    "targets": [[_site(p, SEQUENCE[p]) for p in positions]],
                }
            )
        }
    )
    return tokenize_dataset(
        raw,
        tokenizer=tokenizer,
        task_type="calibration",
        sequence_column="sequence",
        label_column="targets",
        max_length=max_length,
        random_truncate=random_truncate,
    )["test"]


def _consensus(label: list[float]) -> str:
    return CANONICAL_AMINO_ACIDS[int(np.argmax(label[21:41]))]


def test_windows_are_centred_and_clamped(protein_tokenizer):
    rows = _tokenize(protein_tokenizer, max_length=100_000)
    aa_ids = resolve_aa_token_ids(protein_tokenizer)
    residue_ids = protein_tokenizer(SEQUENCE, add_special_tokens=False)["input_ids"]

    assert rows["id"] == [f"P1:{p}" for p in POSITIONS]
    # Expected window starts: centred, then clamped at the N- and C-termini.
    expected_starts = [0, 600 - WINDOW_RESIDUES // 2, len(SEQUENCE) - WINDOW_RESIDUES]
    for row, pos, start in zip(rows, POSITIONS, expected_starts):
        input_ids, label = row["input_ids"], row["targets"]
        mask_idx = int(label[0])
        assert len(input_ids) == WINDOW_RESIDUES + 2
        assert input_ids[0] == protein_tokenizer.bos_token_id
        assert input_ids[-1] == protein_tokenizer.eos_token_id
        assert input_ids[mask_idx] == protein_tokenizer.mask_token_id
        assert mask_idx == 1 + pos - start
        unmasked = list(input_ids)
        unmasked[mask_idx] = residue_ids[pos]
        assert unmasked[1:-1] == residue_ids[start : start + WINDOW_RESIDUES]
        assert [int(v) for v in label[1:21]] == list(aa_ids)
        assert _consensus(label) == SEQUENCE[pos]
        assert [int(v) for v in label[41:]] == SITE


@pytest.mark.parametrize("random_truncate", [False, True])
def test_cropped_sequences_keep_only_in_window_positions(
    protein_tokenizer, random_truncate
):
    rows = _tokenize(
        protein_tokenizer,
        max_length=64,
        random_truncate=random_truncate,
        positions=list(range(0, len(SEQUENCE), 7)),
    )

    # 62 residues survive the crop, i.e. 8 or 9 of the every-7th positions.
    assert 8 <= len(rows) <= 9
    for row in rows:
        pos = int(row["id"].split(":")[1])
        assert _consensus(row["targets"]) == SEQUENCE[pos]
    if not random_truncate:
        assert rows["id"] == [f"P1:{p}" for p in range(0, 62, 7)]


def _profiles(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def test_forward_kl_matches_direct_formula_with_zero_frequencies():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 20))
    profiles = rng.dirichlet(np.full(20, 0.3), size=3)
    profiles[0] = 0.0
    profiles[0, 3] = 1.0
    temperatures = np.array([1.0, 2.0, 0.5])

    model = softmax(logits / temperatures[:, None], axis=1)
    support = profiles > 0
    expected = [
        np.sum(p[s] * np.log(p[s] / q[s])) for p, q, s in zip(profiles, model, support)
    ]
    np.testing.assert_allclose(forward_kl(logits, profiles, temperatures), expected)
    np.testing.assert_allclose(
        forward_kl(logits, _profiles(logits, 1.0), 1.0), 0, atol=1e-12
    )


@pytest.mark.parametrize(
    ("true_temperature", "expected", "bound"),
    [
        (2.0, 2.0, "none"),
        (0.6, 0.6, "none"),
        (20.0, 8.0, "upper"),
        (0.1, 0.25, "lower"),
    ],
)
def test_fit_temperature_recovers_the_profile_temperature(
    true_temperature, expected, bound
):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(36, 20))
    strata = np.arange(36) % 18

    temperature, hit = fit_temperature(
        logits, _profiles(logits, true_temperature), strata
    )

    assert temperature == pytest.approx(expected, rel=1e-6)
    assert hit == bound


def test_crossfit_fits_on_other_folds_balanced_rows_only():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(60, 20))
    folds = np.repeat([0, 1, 2], 20)
    fit_rows = np.arange(60) % 4 != 0  # every 4th row is high-NT only
    # Fold 0 profiles follow T=2, fold 1 T=0.5; fold 2's high-NT-only rows
    # would pull the fit to T=8 if they were used.
    temperature = np.select([folds == 0, folds == 1], [2.0, 0.5], 8.0)
    temperature[(folds == 2) & fit_rows] = 2.0
    profiles = softmax(logits / temperature[:, None], axis=1)

    fitted = crossfit_temperatures(
        logits, profiles, np.arange(60) % 18, folds, fit_rows
    )

    assert fitted[folds == 1] == pytest.approx(2.0, rel=1e-6)
    assert np.unique(fitted[folds == 0]).size == 1
    assert 0.5 < fitted[0] < 2.0


def test_score_crossfits_trained_and_control_temperatures():
    rng = np.random.default_rng(3)
    n = 40
    logits = rng.normal(size=(n, 20))
    profiles = _profiles(logits, 2.0)
    is_t = np.arange(n) % 2
    in_balanced = (np.arange(n) % 5 != 0).astype(int)
    in_high_nt = 1 - in_balanced
    folds = np.arange(n) % 2
    bins = np.arange(n) % 9
    consensus_mask = 1 << profiles.argmax(1)
    labels = np.column_stack(
        [profiles, is_t, bins, folds, in_balanced, in_high_nt, consensus_mask]
    )
    ids = [f"P{i // 4}:{i}" for i in range(n)]
    output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name="toy", dataset_name="toy", task_type="calibration"
        ),
        predictions=pl.DataFrame(
            {"id": ids, "prediction": logits.tolist(), "label": labels.tolist()}
        ),
        controls={
            "random_trunk": pl.DataFrame(
                {"id": ids, "prediction": (3 * logits).tolist()}
            )
        },
        requested_controls=["random_trunk"],
    )
    task = get_task("calibration")

    result = task.score(output, ScoreConfig(metrics=list(task.metrics)), seed=0)

    trained = result.condition_scores["trained"]
    control = result.condition_scores["random_trunk"]
    t_rows = (is_t == 1) & (in_balanced == 1)
    kl = forward_kl(logits, profiles, 1.0)
    assert trained["kl_t"] == pytest.approx(kl[t_rows].mean())
    assert trained["calibration_t"] == pytest.approx(100 * np.exp(-kl[t_rows].mean()))
    assert trained["calibration_fit_high_nt"] == pytest.approx(100)
    assert trained["temperature"] == pytest.approx(2.0)
    assert control["temperature"] == pytest.approx(6.0)
    assert control["calibration_fit_nt"] == pytest.approx(100)
    assert trained["accuracy_t"] == control["accuracy_nt"] == 1.0


def test_bootstrap_resamples_whole_proteins():
    ids = ["P1:3", "P1:10", "P2:0", "sp|P3|X:5"]
    groups = id_groups([get_task("calibration").bootstrap_group_id(i) for i in ids])
    np.testing.assert_array_equal(groups, [0, 0, 1, 2])


def _write_subset(directory, sequences, sites):
    directory.mkdir(parents=True)
    pl.DataFrame(sequences, schema=["id", "sequence"], orient="row").write_csv(
        directory / "sequences.csv"
    )
    positions, support = [], []
    for name, position, wt, profile, category, consensus, proportion in sites:
        freqs = {aa: profile.get(aa, 0.0) for aa in CANONICAL_AMINO_ACIDS}
        positions.append(
            {"sequence_name": name, "position": position, "wt_aa": wt, **freqs}
        )
        support.append(
            {
                "sequence_name": name,
                "position": position,
                "category": category,
                "consensus_aas": consensus,
                "consensus_proportion": proportion,
            }
        )
    pl.DataFrame(positions).write_csv(directory / "consensus_positions.csv")
    pl.DataFrame(support).write_csv(directory / "position_support.csv")


@pytest.fixture
def export_dir(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps({"dataset": "toy", "selection": {"edges": CONSERVATION_EDGES}})
    )
    sequences = [("P1", "ACDE"), ("P2", "MKLV"), ("P3", "ACDE")]
    _write_subset(
        tmp_path / "balanced",
        sequences,
        [
            ("P1", 0, "A", {"A": 0.15, "C": 0.85}, "NT", "C", 0.85),
            ("P2", 1, "K", {"K": 1.0}, "T", "K", 1.0),
            ("P3", 2, "D", {"G": 0.5, "N": 0.5}, "NT", "G|N", 0.5),
        ],
    )
    _write_subset(
        tmp_path / "high_nt_095",
        sequences[:1],
        [
            ("P1", 0, "A", {"A": 0.15, "C": 0.85}, "NT", "C", 0.85),
            ("P1", 3, "E", {"D": 0.96, "E": 0.04}, "NT", "D", 0.96),
        ],
    )
    return tmp_path


def test_sourcing_merges_subsets_into_site_targets(export_dir):
    dataset, sequences, sites = read_export(export_dir)
    folds = assign_folds(sequences)
    frame = build_calibration_frame(sequences, sites, folds)

    assert dataset == "toy"
    fold_of = dict(folds.iter_rows())
    assert fold_of["P1"] == fold_of["P3"]  # identical sequences stay together
    assert frame["id"].to_list() == ["P1", "P2", "P3"]
    targets = {
        (name, int(t[0])): t[21:]
        for name, rows in zip(frame["id"], frame["targets"].to_list())
        for t in rows
    }
    aa = CANONICAL_AMINO_ACIDS.index
    # [is_t, bin, fold, in_balanced, in_high_nt, consensus_mask]
    assert targets[("P1", 0)] == [0, 7, fold_of["P1"], 1, 1, 1 << aa("C")]
    assert targets[("P1", 3)] == [0, 8, fold_of["P1"], 0, 1, 1 << aa("D")]
    assert targets[("P2", 1)] == [1, 8, fold_of["P2"], 1, 0, 1 << aa("K")]
    assert targets[("P3", 2)] == [
        0,
        4,
        fold_of["P3"],
        1,
        0,
        (1 << aa("G")) | (1 << aa("N")),
    ]
    p2 = frame["targets"][1].to_list()[0]
    assert p2[1 + aa("K")] == 1.0


def test_sourcing_rejects_wt_mismatch(export_dir):
    path = export_dir / "balanced" / "consensus_positions.csv"
    path.write_text(path.read_text().replace("P2,1,K", "P2,1,M"))

    with pytest.raises(ValueError, match="wt_aa"):
        read_export(export_dir)
