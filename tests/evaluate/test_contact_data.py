"""HF-only CASP loading, alignment and full-chain pipeline integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from datasets import Dataset, DatasetDict, Features, List, Value

from modules.evaluate.src.dataset import dataloader
from modules.evaluate.src.dataset.dataloader import load_contact_dataset
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.schemas.artifacts import (
    ContactEvaluationMetadata,
    PreparedArtifacts,
)
from modules.evaluate.src.steps.predict import PredictConfig, PredictStep
from modules.evaluate.src.steps.prepare import PrepareConfig, PrepareStep
from modules.evaluate.src.steps.score import ScoreConfig, ScoreStep
from modules.evaluate.src.tasks.contact_prediction import (
    candidate_pairs,
    prepare_contact_dataset,
    tokenize_contact_chain,
)

FEATURES = Features(
    {
        "id": Value("string"),
        "record_id": Value("string"),
        "sequence": Value("string"),
        "split": Value("string"),
        "source": Value("string"),
        "edition": Value("int32"),
        "complex_id": Value("string"),
        "target": Value("string"),
        "unit_id": Value("string"),
        "kind": Value("string"),
        "positions": List(Value("int32")),
        "labels_available": Value("bool"),
        "positive_pairs": List(List(Value("int32"))),
        "valid_positions": List(Value("int32")),
    }
)


def row(record_id="chain", split="test", sequence="A" * 31, **overrides):
    result = {
        "record_id": record_id,
        "sequence": sequence,
        "split": split,
        "source": "casp" if split == "test" else "trrosetta",
        "edition": 14 if split == "test" else None,
        "complex_id": "complex" if split == "test" else None,
        "target": record_id if split == "test" else None,
        "unit_id": "chain",
        "kind": "global",
        "positions": None,
        "labels_available": True,
        "positive_pairs": [[0, 6]] if len(sequence) > 6 else [],
        "valid_positions": list(range(len(sequence))),
    }
    result.update(overrides)
    result["id"] = f"{record_id}/{result['kind']}/{result['unit_id']}"
    return result


def hf_data(rows, *, features=FEATURES):
    return DatasetDict(
        {
            "train": Dataset.from_list([row("train", "train")], features=features),
            "validation": Dataset.from_list(
                [row("validation", "validation")], features=features
            ),
            "test": Dataset.from_list(rows, features=features),
        }
    )


def prediction_inputs(dataset, predictor, tmp_path):
    workspace = get_evaluation_workspace(
        EvaluationWorkspaceConfig(
            model_name="toy",
            dataset_name="casp",
            run_id="offline",
            task_type="contact_prediction",
            target_type="categorical",
            label_format="contact_pairs",
            output_dir=tmp_path,
        ),
        variant="bilinear512-frozen_trunk",
        seed=0,
    )
    prepared = PreparedArtifacts(
        model=SimpleNamespace(predict=lambda chain, tokenizer: predictor(chain)),
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        contact_dataset=dataset,
    )
    return workspace, prepared


def test_local_loading_uses_only_parquet_and_preserves_rows(tmp_path, monkeypatch):
    rows = [
        row(),
        row("edition15", edition=15),
        row(kind="intra_eu", unit_id="D1", positions=[0, 6, 8], valid_positions=[0, 6]),
    ]
    hf = hf_data(rows)
    data = tmp_path / "data"
    data.mkdir()
    for split, dataset in hf.items():
        dataset.to_parquet(data / f"{split}-00000-of-00001.parquet")
    monkeypatch.setattr(np, "load", Mock(side_effect=AssertionError("No coordinates")))
    loaded = PrepareStep(PrepareConfig()).run_contacts(
        tmp_path, cache_dir=tmp_path / "cache"
    )
    expected = prepare_contact_dataset(hf)
    assert loaded.fingerprint == expected.fingerprint
    assert loaded.splits == expected.splits
    assert [len(chain.units) for chain in loaded.splits["test"]] == [2, 1]
    selected = load_contact_dataset(tmp_path, edition=15, cache_dir=tmp_path / "cache")
    assert selected.splits["train"] == loaded.splits["train"]
    assert selected.splits["validation"] == loaded.splits["validation"]
    assert [chain.record_id for chain in selected.splits["test"]] == ["edition15"]
    assert selected.fingerprint != loaded.fingerprint


def test_hub_loading_requires_and_passes_commit_sha(monkeypatch):
    loader = Mock(return_value=hf_data([row()]))
    monkeypatch.setattr(dataloader, "load_dataset", loader)
    for revision in (None, "main", "v6"):
        with pytest.raises(ValueError, match="commit SHA"):
            load_contact_dataset("flair-bio/contact_prediction_casp", revision=revision)
    loader.assert_not_called()
    sha = "a" * 40
    loaded = load_contact_dataset("flair-bio/contact_prediction_casp", revision=sha)
    loader.assert_called_once_with("flair-bio/contact_prediction_casp", revision=sha)
    assert loaded.revision == sha


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"positive_pairs": [[0, 6, 7]]}, "exactly two"),
        ({"positive_pairs": [[6, 0]]}, "strictly increasing"),
        ({"positive_pairs": [[0, 0]]}, "strictly increasing"),
        ({"positive_pairs": [[0, 31]]}, "out-of-bounds"),
        ({"positive_pairs": [[0.5, 6.0]]}, "noninteger"),
        ({"positive_pairs": [[0, 6], [0, 6]]}, "Duplicate positive"),
        ({"valid_positions": [0]}, "two valid"),
        ({"valid_positions": [0, 0, 6]}, "strictly increasing"),
        ({"labels_available": False}, "Unavailable labels"),
        ({"positions": []}, "null positions"),
        ({"sequence": "a" * 31}, "uppercase protein"),
        ({"split": "train"}, "split column"),
        ({"edition": None}, "CASP metadata"),
    ],
)
def test_rejects_malformed_rows(change, message):
    with pytest.raises(ValueError, match=message):
        prepare_contact_dataset(hf_data([row(**change)], features=None))


def test_validates_grouping_and_fixed_splits():
    hf = hf_data([row()])
    del hf["validation"]
    with pytest.raises(ValueError, match="fixed"):
        prepare_contact_dataset(hf)
    with pytest.raises(ValueError, match="Duplicate scoring"):
        prepare_contact_dataset(hf_data([row(), row()]))
    with pytest.raises(ValueError, match="Duplicate scoring|multiple splits"):
        prepare_contact_dataset(hf_data([row("train")]))
    eu = row(kind="intra_eu", unit_id="D1", positions=[0, 6], valid_positions=[0, 6])
    with pytest.raises(ValueError, match="exactly one global"):
        prepare_contact_dataset(hf_data([eu]))
    with pytest.raises(ValueError, match="inconsistent"):
        prepare_contact_dataset(hf_data([row(), dict(eu, complex_id="other")]))
    with pytest.raises(ValueError, match="restricted to EU"):
        prepare_contact_dataset(
            hf_data([row(), dict(eu, valid_positions=list(range(31)))])
        )
    hf = hf_data([row()])
    hf["test"] = hf["test"].remove_columns("positive_pairs")
    with pytest.raises(ValueError, match="missing contact columns"):
        prepare_contact_dataset(hf)


def test_generic_loader_preserves_fixed_splits_but_generic_tokenization_rejects_rows(
    monkeypatch, protein_tokenizer
):
    hf = hf_data([row()])
    monkeypatch.setattr(
        dataloader, "load_raw_evaluation_dataset", Mock(return_value=hf)
    )
    sha = "a" * 40
    assert dataloader.load_evaluation_dataset("repo/data", revision=sha) is hf
    with pytest.raises(ValueError, match="unchanged fixed"):
        dataloader.load_evaluation_dataset(
            "repo/data",
            revision=sha,
            split=dataloader.DatasetSplitConfig(validation=None),
        )
    with pytest.raises(ValueError, match="contact step methods"):
        dataloader.tokenize_dataset(
            hf, protein_tokenizer, "contact_prediction", "sequence", "targets", 512
        )


def test_original_indices_and_special_tokens_survive_cleaning(model_family):
    chain = prepare_contact_dataset(
        hf_data([row(sequence="XABGZ", positive_pairs=[[1, 3]])])
    ).splits["test"][0]
    alignment = chain.alignment
    assert alignment.clean_sequence == "AG"
    assert alignment.retained_positions == (1, 3)
    assert alignment.original_to_retained == (-1, 0, -1, 1, -1)
    tokens = tokenize_contact_chain(chain, model_family.tokenizer)
    assert tokens.residue_token_indices == (1, 2)
    assert tokens.position_ids == (0, 2, 4, 6)
    pairs = candidate_pairs(chain.units[0], alignment, minsep=2)
    assert pairs.i.tolist() == [0] and pairs.j.tolist() == [1]
    assert pairs.labels.tolist() == [1]
    assert pairs.original_i.tolist() == [1] and pairs.original_j.tolist() == [3]
    assert not len(candidate_pairs(chain.units[0], alignment).i)


def test_full_chain_tokenization_ignores_default_limits(model_family):
    chain = prepare_contact_dataset(hf_data([row(sequence="A" * 3364)])).splits["test"][
        0
    ]
    tokens = tokenize_contact_chain(chain, model_family.tokenizer)
    assert len(tokens.input_ids) == 3366
    assert tokens.residue_token_indices[-1] == 3364
    assert tokens.position_ids[-1] == 3365


def test_tokenizer_that_drops_residues_is_rejected(protein_tokenizer):
    chain = prepare_contact_dataset(hf_data([row()])).splits["test"][0]
    tokenizer = Mock()
    encoded = protein_tokenizer(
        chain.alignment.clean_sequence, return_special_tokens_mask=True
    )
    tokenizer.return_value = {key: values[:-2] for key, values in encoded.items()}
    tokenizer.convert_tokens_to_ids = protein_tokenizer.convert_tokens_to_ids
    tokenizer.unk_token_id = protein_tokenizer.unk_token_id
    with pytest.raises(ValueError, match="one unchanged token"):
        tokenize_contact_chain(chain, tokenizer)


def test_eu_length_includes_unknown_members_but_not_deleted_residues():
    sequence = "AXAAAAXAAA"
    eu = row(
        sequence=sequence,
        kind="intra_eu",
        unit_id="D1",
        positions=[0, 1, 5, 7, 9],
        valid_positions=[0, 5, 9],
        positive_pairs=[[0, 9]],
    )
    chain = prepare_contact_dataset(
        hf_data([row(sequence=sequence, positive_pairs=[]), eu])
    ).splits["test"][0]
    pairs = candidate_pairs(chain.units[1], chain.alignment)
    assert pairs.length == 4  # Member 7 is retained but structurally unknown.
    assert list(zip(pairs.original_i, pairs.original_j)) == [(0, 9)]
    assert pairs.labels.tolist() == [1]


def test_steps_share_full_chain_predictions_and_roundtrip_outputs(tmp_path):
    missing = {"labels_available": False, "positive_pairs": [], "valid_positions": []}
    rows = [
        row(**missing),
        row(
            kind="intra_eu",
            unit_id="D1",
            positions=list(range(10)),
            valid_positions=list(range(10)),
        ),
        row("duplicate"),
        row("unknown", sequence="G" * 40, **missing),
        row(
            kind="intra_eu",
            unit_id="empty",
            positions=[],
            valid_positions=[],
            positive_pairs=[],
        ),
    ]
    dataset = prepare_contact_dataset(hf_data(rows))
    predictor = Mock(
        side_effect=lambda chain: np.zeros((len(chain.alignment.clean_sequence),) * 2)
    )
    workspace, prepared = prediction_inputs(dataset, predictor, tmp_path)
    predictions = PredictStep(PredictConfig(controls=[])).run(
        workspace, prepared, seed=0
    )
    predictor.assert_called_once_with(dataset.splits["test"][0])
    assert predictions.predictions.height == len(rows)
    assert predictions.predictions_path is not None
    restored = ScoreStep.load_predict_output(workspace)
    assert isinstance(restored.metadata, ContactEvaluationMetadata)
    assert restored.metadata == predictions.metadata
    scored = ScoreStep(ScoreConfig()).run_contacts(restored, output_dir=tmp_path)
    assert scored.unit_scores.height == 2 * len(rows)
    statuses = {
        (r["id"], r["minsep"]): r["status"] for r in scored.unit_scores.to_dicts()
    }
    assert statuses["chain/global/chain", 6] == "missing_labels"
    assert statuses["unknown/global/chain", 24] == "missing_labels"
    assert statuses["chain/intra_eu/D1", 6] == "scored"
    assert statuses["chain/intra_eu/D1", 24] == "no_eligible_pairs"
    assert statuses["chain/intra_eu/empty", 6] == "no_retained_members"
    assert scored.unit_scores_path.exists() and scored.summaries_path.exists()
    assert pl.read_parquet(scored.unit_scores_path).equals(scored.unit_scores)


def test_deleted_chains_are_unscorable_and_bad_predictions_fail(tmp_path):
    step = PredictStep(PredictConfig(controls=[], persist_to_disk=False))
    dataset = prepare_contact_dataset(
        hf_data([row(sequence="XBOUZJ", positive_pairs=[])])
    )
    predictor = Mock(side_effect=AssertionError("No retained residues"))
    workspace, prepared = prediction_inputs(dataset, predictor, tmp_path)
    predictions = step.run(workspace, prepared, seed=0)
    predictor.assert_not_called()
    result = ScoreStep(ScoreConfig()).run_contacts(predictions)
    assert result.unit_scores["status"].to_list() == ["no_retained_members"] * 2
    dataset = prepare_contact_dataset(hf_data([row()]))
    workspace, prepared = prediction_inputs(
        dataset, lambda chain: np.zeros((10, 10)), tmp_path
    )
    with pytest.raises(ValueError, match="full retained-chain"):
        step.run(workspace, prepared, seed=0)
    workspace, prepared = prediction_inputs(
        dataset, Mock(side_effect=MemoryError("too long")), tmp_path
    )
    with pytest.raises(MemoryError, match="too long"):
        step.run(workspace, prepared, seed=0)
    with pytest.raises(ValueError, match="fitted predictors"):
        PredictStep(PredictConfig()).run(workspace, prepared, seed=0)
    with pytest.raises(ValueError, match="complexes"):
        ScoreStep(ScoreConfig(bootstrap_enabled=True)).run_contacts(predictions)
