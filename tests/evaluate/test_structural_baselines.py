"""Baseline-only contact separation prior and amino-acid frequency (PPL) baselines.

All inputs are tiny synthetic structures; no Hub access happens here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from modules.evaluate.src.predictors.composition import CANONICAL_AMINO_ACIDS
from modules.evaluate.src.predictors.priors import (
    MIN_SEPARATION,
    fit_amino_acid_frequencies,
)
from modules.evaluate.src.runners import structural_baselines as sb
from modules.evaluate.src.tasks.contact_prediction import (
    ContactChain,
    ContactDataset,
    ContactUnit,
    align_contact_sequence,
)
from modules.evaluate.src.utils.baselines import (
    AminoAcidReference,
    BaselinesConfig,
)

SHA = "258a928947a5232be1a6292b0fb9466ec4dada8c"
DATASET_IDENTITY = {"source": "flair-bio/contact_prediction_casp", "revision": SHA}


def _unit(record_id: str, sequence: str, pairs, *, labels=True, valid=None):
    length = len(sequence)
    positions = tuple(range(length)) if valid is None else tuple(valid)
    return ContactUnit(
        id=f"{record_id}/global/chain",
        unit_id="chain",
        kind="global",
        positions=None,
        labels_available=labels,
        positive_pairs=tuple(pairs) if labels else (),
        valid_positions=positions if labels else (),
    )


def _chain(record_id: str, split: str, sequence: str, units):
    return ContactChain(
        record_id=record_id,
        split=split,
        source="trrosetta" if split == "train" else "casp",
        edition=None if split == "train" else 14,
        complex_id=None if split == "train" else f"complex-{record_id}",
        target=None,
        alignment=align_contact_sequence(sequence),
        units=tuple(units),
    )


def _dataset(train_chains, test_chains=()):
    return ContactDataset(
        splits={"train": tuple(train_chains), "test": tuple(test_chains)},
        fingerprint="fp",
        source="flair-bio/contact_prediction_casp",
        revision=SHA,
    )


# ----------------------------------------------------------------- configuration


def test_task_routes_accept_only_their_declared_baselines():
    contact = BaselinesConfig(names=["separation_prior"])
    contact.validate_task("contact_prediction")
    with pytest.raises(
        ValueError, match="requires task_type='sequence_classification'"
    ):
        BaselinesConfig(names=["composition"]).validate_task("contact_prediction")

    reference = AminoAcidReference(repo_id="o/r", revision=SHA)
    BaselinesConfig(
        names=["amino_acid_frequency"], amino_acid_reference=reference
    ).validate_task("pseudo_perplexity")
    with pytest.raises(ValueError, match="requires task_type='contact_prediction'"):
        BaselinesConfig(
            names=["separation_prior"], amino_acid_reference=reference
        ).validate_task("pseudo_perplexity")


def test_multilabel_baselines_are_rejected():
    with pytest.raises(ValueError, match="not supported"):
        BaselinesConfig(names=["majority_class"]).validate_task("multilabel")


def test_amino_acid_frequency_requires_an_explicit_reference():
    with pytest.raises(ValueError, match="amino_acid_reference"):
        BaselinesConfig(names=["amino_acid_frequency"]).validate_task(
            "pseudo_perplexity"
        )


@pytest.mark.parametrize("revision", ["main", "258a9289", None])
def test_reference_requires_a_full_commit_sha(revision):
    with pytest.raises(ValidationError):
        AminoAcidReference(repo_id="o/r", revision=revision)


def test_reference_split_must_be_train():
    with pytest.raises(ValidationError):
        AminoAcidReference(repo_id="o/r", revision=SHA, split="test")


def test_reference_loader_reads_only_train_at_pinned_revision():
    calls = []

    def loader(repo_id, *, split, revision):
        calls.append((repo_id, split, revision))
        return [{"sequence": "ACD"}, {"sequence": "EF"}]

    reference = AminoAcidReference(repo_id="o/r", revision=SHA)
    assert sb.load_amino_acid_reference(reference, loader=loader) == ["ACD", "EF"]
    assert calls == [("o/r", "train", SHA)]


# ------------------------------------------------------------ separation prior


def test_training_pairs_dedupe_canonical_ids_and_respect_min_separation():
    sequence = "ACDEFGHIKL"
    pairs = [(0, 6), (0, 3), (1, 9)]
    # The same canonical protein appears twice (duplicate record); the pair must
    # count once, and a contact in either copy marks it positive.
    train = [
        _chain("a", "train", sequence, [_unit("a", sequence, pairs)]),
        _chain("a2", "train", sequence, [_unit("a2", sequence, [(0, 6)])]),
    ]
    rows = sb.separation_training_pairs(_dataset(train))
    # L=10 has 10 canonical pairs with separation >= 6 (i=0:4, 1:3, 2:2, 3:1).
    # Duplicate chains collapse to these 10 IDs, not 10 + 10.
    assert len(rows) == 10
    assert all(separation >= MIN_SEPARATION for separation, _ in rows)
    # Positives are exactly (0,6) and (1,9); the sep-3 pair never enters.
    assert sorted(s for s, is_contact in rows if is_contact) == [6, 8]


def test_unlabelled_units_and_test_chains_do_not_train_the_prior():
    sequence = "ACDEFGHIKL"
    train = [_chain("a", "train", sequence, [_unit("a", sequence, [], labels=False)])]
    test = [_chain("t", "test", sequence, [_unit("t", sequence, [(0, 6)])])]
    assert sb.separation_training_pairs(_dataset(train, test)) == []


def test_separation_matrix_uses_original_positions_and_zeroes_short_range():
    # 'X' is non-canonical and removed from the retained order, so retained
    # positions have gaps; separations must use original coordinates.
    sequence = "ACXDEFGHIKLMNP"
    alignment = align_contact_sequence(sequence)
    from modules.evaluate.src.predictors.priors import fit_separation_prior

    prior = fit_separation_prior([(6, True), (6, False), (9, True)])
    matrix = sb.separation_matrix(prior, alignment)
    retained = np.asarray(alignment.retained_positions)
    assert matrix.shape == (len(retained), len(retained))
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0)
    short = np.abs(retained[:, None] - retained[None, :]) < MIN_SEPARATION
    assert np.all(matrix[short] == 0)
    # Original positions 0 and 6 (retained index 0 and 5 due to the gap) get the
    # separation-6 probability, not a retained-index separation of 5.
    first, sixth = 0, int(np.where(retained == 6)[0][0])
    assert matrix[first, sixth] == pytest.approx(prior.probability(6))


def _run_contact(tmp_path, dataset, seed=7):
    return sb.run_contact_separation_baseline(
        dataset=dataset,
        dataset_identity=DATASET_IDENTITY,
        root=tmp_path,
        seed=seed,
    )


def _synthetic_contact_dataset(*, test_labels=((0, 6),)):
    train_seq = "ACDEFGHIKLMNPQRSTVWY" * 2
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    train = [
        _chain(
            "train0",
            "train",
            train_seq,
            [_unit("train0", train_seq, [(0, 6), (2, 12), (5, 30)])],
        )
    ]
    test = [
        _chain(
            "test0",
            "test",
            test_seq,
            [_unit("test0", test_seq, list(test_labels))],
        )
    ]
    return _dataset(train, test)


def test_contact_baseline_writes_casp_scores_and_manifest(tmp_path):
    path = _run_contact(tmp_path, _synthetic_contact_dataset())
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["baseline"] == "separation_prior"
    assert manifest["identity"]["protocol"]["min_separation"] == MIN_SEPARATION
    assert (path / "unit_scores.parquet").exists()
    summaries = pl.read_parquet(path / "summaries.parquet")
    assert summaries.height > 0


def test_contact_baseline_run_once_reuses_identical_artifact(tmp_path, monkeypatch):
    dataset = _synthetic_contact_dataset()
    first = _run_contact(tmp_path, dataset)
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise AssertionError("baseline must not be recomputed")

    monkeypatch.setattr(sb, "_compute_contact_scores", boom)
    assert _run_contact(tmp_path, dataset) == first
    assert calls == []


def test_contact_fit_ignores_test_labels(tmp_path):
    base = _run_contact(tmp_path / "a", _synthetic_contact_dataset())
    changed = _run_contact(
        tmp_path / "b", _synthetic_contact_dataset(test_labels=((1, 9), (4, 20)))
    )
    base_manifest = json.loads((base / "manifest.json").read_text())
    changed_manifest = json.loads((changed / "manifest.json").read_text())
    assert base_manifest["identity"]["fit"] == changed_manifest["identity"]["fit"], (
        "test labels must not change the training fit identity"
    )
    assert base != changed  # but the scored test artifact is distinct


def test_tampered_contact_manifest_is_rejected(tmp_path):
    path = _run_contact(tmp_path, _synthetic_contact_dataset())
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["identity"]["fit"]["dataset"] = {"source": "other"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incompatible|Corrupt"):
        _run_contact(tmp_path, _synthetic_contact_dataset())


# ---------------------------------------------------- amino-acid frequency (PPL)


def test_amino_acid_baseline_fits_reference_and_excludes_non_canonical(tmp_path):
    reference = ["ACDEFGHIKLMNPQRSTVWY", "AAAA"]
    frequencies = fit_amino_acid_frequencies(reference)
    tests = [("p1", "ACDX"), ("p2", "AAB")]
    path = sb.run_amino_acid_frequency_baseline(
        reference=AminoAcidReference(repo_id="o/r", revision=SHA),
        reference_sequences=reference,
        test_rows=tests,
        dataset_identity={"source": "flair-bio/ppl"},
        root=tmp_path,
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["metrics"]["excluded_residues"] == 2
    assert manifest["metrics"]["residues"] == 5
    logp = [math.log(frequencies[CANONICAL_AMINO_ACIDS.index(c)]) for c in "ACDAA"]
    expected = math.exp(-sum(logp) / len(logp))
    assert manifest["metrics"]["pseudo_perplexity"] == pytest.approx(expected)
    frame = pl.read_parquet(path / "predictions.parquet")
    assert frame.columns == ["id", "label", "prediction"]
    assert frame.height == 5


def test_amino_acid_fit_identity_ignores_test_sequences(tmp_path):
    reference = ["ACDEFG", "HIKLMN"]
    first = sb.run_amino_acid_frequency_baseline(
        reference=AminoAcidReference(repo_id="o/r", revision=SHA),
        reference_sequences=reference,
        test_rows=[("p1", "AAAA")],
        dataset_identity={"source": "d"},
        root=tmp_path,
    )
    second = sb.run_amino_acid_frequency_baseline(
        reference=AminoAcidReference(repo_id="o/r", revision=SHA),
        reference_sequences=reference,
        test_rows=[("p1", "WWWW")],
        dataset_identity={"source": "d"},
        root=tmp_path,
    )
    fit_one = json.loads((first / "manifest.json").read_text())["identity"]["fit"]
    fit_two = json.loads((second / "manifest.json").read_text())["identity"]["fit"]
    assert fit_one == fit_two
    assert first != second


def test_amino_acid_baseline_run_once(tmp_path, monkeypatch):
    kwargs = dict(
        reference=AminoAcidReference(repo_id="o/r", revision=SHA),
        reference_sequences=["ACDE"],
        test_rows=[("p1", "AC")],
        dataset_identity={"source": "d"},
        root=tmp_path,
    )
    first = sb.run_amino_acid_frequency_baseline(**kwargs)

    def boom(*args, **kw):
        raise AssertionError("must not recompute")

    monkeypatch.setattr(sb, "_compute_amino_acid_rows", boom)
    assert sb.run_amino_acid_frequency_baseline(**kwargs) == first


def test_amino_acid_baseline_requires_reference(tmp_path):
    with pytest.raises(ValueError, match="amino_acid_reference"):
        sb.run_amino_acid_frequency_baseline(
            reference=None,
            reference_sequences=[],
            test_rows=[("p", "AC")],
            dataset_identity={"source": "d"},
            root=tmp_path,
        )
