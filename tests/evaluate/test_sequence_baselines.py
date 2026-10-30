"""Composition and MMseqs k-NN sequence baselines (approved conservative protocols)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from modules.evaluate.src.predictors import composition, mmseqs_knn, priors
from modules.evaluate.src.utils import baselines as baseline_utils
from modules.evaluate.src.utils.baselines import (
    BaselinesConfig,
    artifact_directory,
    build_sequence_baseline,
)


# --- composition ----------------------------------------------------------


def test_composition_uses_20_canonical_residues_and_excludes_ambiguous():
    features = composition.composition_features(["AAC", "AXXB", "ZZ"])
    assert features.shape == (3, 20)
    alphabet = composition.CANONICAL_AMINO_ACIDS
    assert len(alphabet) == 20
    a, c = alphabet.index("A"), alphabet.index("C")
    # 'X' and 'B' are excluded from the denominator: 'A' is 1 of 1 valid residue.
    assert features[1, a] == pytest.approx(1.0)
    assert features[0, a] == pytest.approx(2 / 3)
    assert features[0, c] == pytest.approx(1 / 3)
    np.testing.assert_allclose(features[0].sum(), 1.0)
    # A sequence with no canonical residues has an all-zero profile.
    np.testing.assert_array_equal(features[2], np.zeros(20))


def test_composition_is_case_insensitive_and_deterministic():
    first = composition.composition_features(["acd"])
    second = composition.composition_features(["ACD"])
    np.testing.assert_array_equal(first, second)


def test_composition_classifier_is_regularized_multinomial_and_train_only():
    train_sequences = ["AAAA", "AAAC", "KKKK", "KKKR", "WWWW", "WWWY"]
    labels = [0, 0, 1, 1, 2, 2]
    model = composition.fit_composition_classifier(
        train_sequences, labels, num_labels=3
    )
    assert model.C == composition.COMPOSITION_PROTOCOL["classification"]["C"]
    assert (
        model.max_iter == composition.COMPOSITION_PROTOCOL["classification"]["max_iter"]
    )
    prediction, probabilities = composition.predict_composition_classifier(
        model, ["AAAA", "KKKK"], num_labels=3
    )
    assert prediction.tolist() == [0, 1]
    assert probabilities.shape == (2, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_composition_classifier_maps_absent_classes_to_zero_probability():
    model = composition.fit_composition_classifier(
        ["AAAA", "KKKK", "AAAC"], [0, 2, 0], num_labels=4
    )
    _, probabilities = composition.predict_composition_classifier(
        model, ["AAAA"], num_labels=4
    )
    assert probabilities.shape == (1, 4)
    assert probabilities[0, 1] == 0.0 and probabilities[0, 3] == 0.0


def test_composition_classifier_with_single_observed_class_is_constant():
    model = composition.fit_composition_classifier(["AAA", "CCC"], [1, 1], num_labels=3)
    prediction, probabilities = composition.predict_composition_classifier(
        model, ["WWW"], num_labels=3
    )
    assert prediction.tolist() == [1]
    assert probabilities.tolist() == [[0.0, 1.0, 0.0]]


def test_composition_regressor_is_ridge_and_deterministic():
    train_sequences = ["AAAA", "KKKK", "WWWW"]
    targets = [1.0, 5.0, 9.0]
    first = composition.fit_composition_regressor(train_sequences, targets)
    second = composition.fit_composition_regressor(train_sequences, targets)
    assert first.alpha == composition.COMPOSITION_PROTOCOL["regression"]["alpha"]
    np.testing.assert_array_equal(
        composition.predict_composition_regressor(first, ["AAAA"]),
        composition.predict_composition_regressor(second, ["AAAA"]),
    )


# --- MMseqs k-NN ----------------------------------------------------------


def test_mmseqs_protocol_is_fixed_and_documented():
    protocol = mmseqs_knn.MMSEQS_KNN_PROTOCOL
    assert protocol["k"] == 5
    assert protocol["sensitivity"] == 7.5
    assert protocol["evalue"] == 1e-3
    assert protocol["max_seqs"] == 300


def test_parse_search_hits_reads_tabular_output():
    text = "q0\tt3\t120\t1e-30\nq0\tt1\t80\t1e-10\nq1\tt2\t50\t1e-4\n"
    hits = mmseqs_knn.parse_search_hits(text)
    assert [h.target for h in hits[0]] == [3, 1]
    assert hits[0][0].bits == 120.0
    assert hits[1][0].target == 2


def test_select_neighbors_is_deterministic_with_dedup_and_tiebreak():
    hits = {
        0: [
            mmseqs_knn.Hit(target=2, bits=100.0, evalue=1e-5),
            mmseqs_knn.Hit(target=0, bits=100.0, evalue=1e-5),
            mmseqs_knn.Hit(target=2, bits=90.0, evalue=1e-4),  # duplicate target
            mmseqs_knn.Hit(target=5, bits=10.0, evalue=1e-2),
        ]
    }
    selected = mmseqs_knn.select_neighbors(hits, num_queries=2, k=2)
    # Ties on bits and evalue go to the lower target index; duplicates removed.
    assert [h.target for h in selected[0]] == [0, 2]
    assert selected[1] == []


def test_knn_classification_votes_by_bits_with_lowest_class_tie_break():
    neighbors = {
        0: [
            mmseqs_knn.Hit(target=0, bits=10.0, evalue=1e-5),
            mmseqs_knn.Hit(target=1, bits=10.0, evalue=1e-5),
        ],
    }
    # Training prior: class 2 occurs twice, class 1 once.
    labels = [2, 1, 2]
    prediction, probabilities = mmseqs_knn.knn_classify(
        neighbors,
        train_labels=labels,
        num_queries=2,
        num_labels=3,
        k=5,
    )
    # Equal bit weight for class 1 and 2: the lower class index wins.
    assert prediction[0] == 1
    np.testing.assert_allclose(probabilities[0], [0.0, 0.5, 0.5])
    # No hit: the training class prior (train-only) is used.
    assert prediction[1] == 2
    np.testing.assert_allclose(probabilities[1], [0.0, 1 / 3, 2 / 3])


def test_knn_regression_is_bit_weighted_mean_with_train_mean_fallback():
    neighbors = {
        0: [
            mmseqs_knn.Hit(target=0, bits=30.0, evalue=1e-5),
            mmseqs_knn.Hit(target=1, bits=10.0, evalue=1e-5),
        ]
    }
    predictions = mmseqs_knn.knn_regress(
        neighbors, train_labels=[4.0, 8.0], num_queries=2
    )
    assert predictions[0] == pytest.approx((4.0 * 30 + 8.0 * 10) / 40)
    assert predictions[1] == pytest.approx(6.0)


def test_search_command_uses_fixed_settings_and_argument_lists(tmp_path):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        # Write the expected tabular output where the command asks for it.
        # Positional layout: mmseqs easy-search QUERY TARGET OUT TMP [options].
        out = argv[4]
        with open(out, "w") as handle:
            handle.write("q0\tt0\t42\t1e-6\n")
        return subprocess.CompletedProcess(argv, 0, "", "")

    with (
        patch.object(mmseqs_knn.shutil, "which", return_value="/bin/mmseqs"),
        patch.object(mmseqs_knn.subprocess, "run", side_effect=fake_run),
    ):
        hits = mmseqs_knn.run_search(["AAAA"], ["AAAA"], work_root=tmp_path)
    argv, kwargs = calls[0]
    assert argv[0] == "/bin/mmseqs"
    assert argv[1] == "easy-search"
    assert "-s" in argv and argv[argv.index("-s") + 1] == "7.5"
    assert "-e" in argv and argv[argv.index("-e") + 1] == "0.001"
    assert "--max-seqs" in argv and argv[argv.index("--max-seqs") + 1] == "300"
    assert kwargs.get("check") is True
    assert kwargs.get("shell", False) is False
    assert hits[0][0].target == 0


def test_missing_mmseqs_binary_is_an_explicit_error(tmp_path):
    with patch.object(mmseqs_knn.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match="mmseqs"):
            mmseqs_knn.run_search(["AAAA"], ["AAAA"], work_root=tmp_path)


def test_real_mmseqs_search_recovers_identical_sequence(tmp_path):
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs binary is not on PATH")
    hits = mmseqs_knn.run_search(
        [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAEKDD"
        ],
        [
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAEKDD"
        ],
        work_root=tmp_path,
    )
    assert 0 in hits and hits[0][0].target == 0


# --- shared-baseline routing ---------------------------------------------


@pytest.mark.parametrize("name", ["composition", "mmseqs_knn"])
def test_new_sequence_baselines_route_to_sequence_tasks(name):
    BaselinesConfig(names=[name]).validate_task("sequence_classification")
    BaselinesConfig(names=[name]).validate_task("sequence_regression")


@pytest.mark.parametrize(
    "task_type",
    ["token_classification", "sequence_multilabel_classification"],
)
def test_unsupported_suites_error_clearly(task_type):
    with pytest.raises(ValueError, match="requires task_type|not supported"):
        BaselinesConfig(names=["composition"]).validate_task(task_type)


def test_protocol_identity_changes_artifact_directory():
    context = {
        "fit": {"v": 1},
        "test": {"v": 1},
        "root": "/tmp/root",
    }
    base = artifact_directory(context, "composition")
    assert baseline_utils.artifact_identity(context, "composition")["protocol"] == (
        composition.COMPOSITION_PROTOCOL
    )
    with patch.dict(
        composition.COMPOSITION_PROTOCOL["classification"], {"C": 99.0}, clear=False
    ):
        assert artifact_directory(context, "composition") != base


def test_build_composition_classification_artifact_columns():
    train = [
        {"id": 0, "sequence": "AAAA", "label": 0},
        {"id": 1, "sequence": "KKKK", "label": 1},
    ]
    test = [{"id": 2, "sequence": "AAAC", "label": 0}]
    frame = build_sequence_baseline("composition", train, test, num_labels=2)
    assert {"id", "label", "prediction", "probability"} <= set(frame.columns)
    assert frame.height == 1
    assert frame["prediction"][0] == 0


def test_build_knn_regression_artifact_has_no_probability():
    train = [
        {"id": 0, "sequence": "AAAA", "label": 1.0},
        {"id": 1, "sequence": "KKKK", "label": 3.0},
    ]
    test = [{"id": 2, "sequence": "AAAA", "label": 0.0}]
    with patch.object(
        mmseqs_knn,
        "run_search",
        return_value={0: [mmseqs_knn.Hit(target=0, bits=50.0, evalue=1e-9)]},
    ):
        frame = build_sequence_baseline("mmseqs_knn", train, test, num_labels=None)
    assert "probability" not in frame.columns
    assert frame["prediction"][0] == pytest.approx(1.0)


# --- reference priors (pseudo-perplexity and contact baselines) -----------


def test_amino_acid_frequencies_are_smoothed_train_reference_only():
    freqs = priors.fit_amino_acid_frequencies(["AAAX", "CC"])
    alphabet = composition.CANONICAL_AMINO_ACIDS
    assert freqs.shape == (20,)
    np.testing.assert_allclose(freqs.sum(), 1.0)
    # Laplace pseudocount 1: A has 3 + 1 of 5 + 20 counts; X is excluded.
    assert freqs[alphabet.index("A")] == pytest.approx(4 / 25)
    assert freqs[alphabet.index("C")] == pytest.approx(3 / 25)
    # Unseen residues keep a strictly positive probability (zero-count policy).
    assert freqs[alphabet.index("W")] == pytest.approx(1 / 25)
    assert priors.amino_acid_log_probability(freqs, "a") == pytest.approx(
        float(np.log(4 / 25))
    )


@pytest.mark.parametrize("residue", ["X", "", "AA"])
def test_amino_acid_log_probability_rejects_non_canonical(residue):
    freqs = priors.fit_amino_acid_frequencies(["ACD"])
    with pytest.raises(ValueError, match="non-canonical"):
        priors.amino_acid_log_probability(freqs, residue)


def test_separation_prior_ignores_short_range_and_smooths_unseen():
    pairs = [(6, True), (6, False), (9, True), (2, True), (2, True)]
    prior = priors.fit_separation_prior(pairs)
    # Separation 6: 1 contact of 2 pairs -> (1+1)/(2+2)
    assert prior.probability(6) == pytest.approx(0.5)
    assert prior.probability(9) == pytest.approx(2 / 3)
    # Separations below 6 never enter the training counts.
    assert prior.pooled == pytest.approx((2 + 1) / (3 + 2))
    # Unseen separation uses the pooled training rate.
    assert prior.probability(30) == pytest.approx(prior.pooled)
    with pytest.raises(ValueError, match="outside the scored range"):
        prior.probability(2)


@pytest.mark.parametrize(
    "preset,task,dataset",
    [
        ("composition", "sequence_classification", "solubility_prediction"),
        ("mmseqs_knn", "sequence_classification", "solubility_prediction"),
    ],
)
def test_new_presets_load_for_sequence_suites(preset, task, dataset):
    from modules.evaluate.src.runners import baselines as runner

    root = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    config = runner.load_baseline_config(
        [
            root / "config.yaml",
            root / f"tasks/{task}.yaml",
            root / f"datasets/{dataset}.yaml",
            root / f"baselines/{preset}.yaml",
        ]
    )
    assert config.baselines.names == [preset]
