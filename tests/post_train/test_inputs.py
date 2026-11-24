import json
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from pydantic import ValidationError

from modules.core.utils.config_loader import load_and_parse
from modules.post_train.src.config import DatasetConfig, MLMConfig, PostTrainingConfig
from modules.post_train.src.dataset.collator import collate_records
from modules.post_train.src.dataset.manifest import load_dataset, load_embeddings, load_manifest


@pytest.mark.parametrize("changes", [
    {"batch_size": 65}, {"microbatch_size": 3},
])
def test_invalid_training_config(changes):
    with pytest.raises(ValidationError):
        PostTrainingConfig(
            student={"checkpoint": "/unused/student"},
            dataset={"path": "/unused/data"},
            output_dir="/unused/output", training=changes,
        )


def test_invalid_mlm_replacement_probabilities():
    with pytest.raises(ValidationError):
        PostTrainingConfig.model_validate({
            "student": {"checkpoint": "/unused/student"},
            "dataset": {"path": "/unused/data"},
            "output_dir": "/unused/output",
            "mlm": {"mask_replace_probability": 0.8, "random_replace_probability": 0.3},
        })


def test_rejects_manifest_outside_dataset():
    with pytest.raises(ValidationError):
        DatasetConfig(path="org/data", manifest_file="../data.json")


def test_recipe_uses_one_full_batch_pool():
    recipe = load_and_parse(
        "modules/post_train/configs/latent_alignment.yaml", PostTrainingConfig,
    )
    assert recipe.alignment.pool_proteins == recipe.training.batch_size


def test_yaml_accepts_nondefault_pool_and_recipe():
    config = load_and_parse(
        "modules/post_train/configs/latent_alignment.yaml", PostTrainingConfig,
        overrides=[
            "alignment.pool_proteins=12", "alignment.projection_dim=24",
            "alignment.loss_weight=0.2", "alignment.initial_temperature=0.2",
            "mlm.loss_weight=0.25",
            "training.batch_size=24", "training.microbatch_size=3", "training.device=cpu",
            "training.betas=[0.8,0.9]", "training.gradient_clip=0.3",
            "seed=123",
        ],
    )
    assert config.alignment.pool_proteins == 12
    assert config.training.betas == (0.8, 0.9)
    assert config.training.device == "cpu"
    assert config.seed == 123
    assert config.mlm.loss_weight == 0.25


def test_residue_mapping_and_corruption(prepared_factory, vocabulary):
    manifest, _, arrays = prepared_factory(dimension=1024)
    prepared = load_manifest(manifest)
    batch = collate_records(prepared.train + prepared.validation, torch.Generator().manual_seed(7), "cpu", vocabulary=vocabulary)
    assert batch["residue_mask"].sum() == 195
    assert batch["attention_mask"].sum() == 201
    torch.testing.assert_close(batch["embeddings"], torch.from_numpy(np.concatenate(arrays)))
    assert (batch["mlm_labels"][~batch["residue_mask"]] == -100).all()
    chosen = batch["mlm_labels"] != -100
    assert chosen.any()
    assert torch.equal(batch["mlm_labels"][chosen], batch["clean_input_ids"][chosen])
    assert (batch["clean_input_ids"][:, 0] == 3).all()
    assert not batch["attention_mask"][0, -1]


def test_random_pad_remains_attended(prepared_factory, vocabulary):
    manifest, _, _ = prepared_factory()
    prepared = load_manifest(manifest)
    found = False
    for seed in range(100):
        batch = collate_records(prepared.train, torch.Generator().manual_seed(seed), "cpu", vocabulary=vocabulary)
        random_pad = (batch["input_ids"] == 0) & batch["residue_mask"]
        if random_pad.any():
            assert batch["attention_mask"][random_pad].all()
            assert (batch["mlm_labels"][random_pad] != -100).all()
            found = True
            break
    assert found


def test_configured_corruption_can_mask_all_residues(prepared_factory, vocabulary):
    manifest, _, _ = prepared_factory()
    prepared = load_manifest(manifest)
    batch = collate_records(
        prepared.train, torch.Generator().manual_seed(7), "cpu",
        vocabulary=vocabulary,
        mlm=MLMConfig(
            mask_probability=1.0, mask_replace_probability=1.0,
            random_replace_probability=0.0,
        ),
    )
    residues = batch["residue_mask"]
    assert torch.equal(batch["mlm_labels"] != -100, residues)
    assert (batch["input_ids"][residues] == 2).all()


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["metadata"].update(embedding_format="other"), "embedding_format"),
    (lambda p: p["metadata"]["embedding_schema"].update(feature_columns=["x", "x"]), "feature columns"),
    (lambda p: p["metadata"]["structural_teacher"].update(dimension=512), "dimension"),
    (lambda p: p["records"][1].update(id=p["records"][0]["id"]), "Duplicate"),
    (lambda p: p["records"][1].update(sequence=p["records"][0]["sequence"],
                                     residue_ids=p["records"][0]["residue_ids"]), "disjoint"),
    (lambda p: p["records"][0].update(embedding_file="../outside.parquet"), "within"),
    (lambda p: p["records"][0].update(residue_ids=["1:"] * 64), "identities"),
])
def test_reject_manifest_mismatch(prepared_factory, mutation, match):
    manifest, payload, _ = prepared_factory()
    mutation(payload)
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        load_manifest(manifest)


def test_data_drives_dimension_and_lengths_with_optional_teacher_provenance(
    prepared_factory, vocabulary,
):
    manifest, payload, arrays = prepared_factory(
        dimension=11, lengths=(3, 386, 5), teacher=None,
        columns=[f"latent_{index}" for index in range(11)],
    )
    prepared = load_manifest(manifest)
    assert prepared.metadata == payload["metadata"]
    batch = collate_records(
        prepared.train + prepared.validation, torch.Generator().manual_seed(7),
        "cpu", vocabulary=vocabulary,
    )
    torch.testing.assert_close(batch["embeddings"], torch.from_numpy(np.concatenate(arrays)))
    assert batch["residue_mask"].sum() == 394
    assert batch["clean_input_ids"].shape == (3, 388)


def test_parquet_dimension_must_match_declared_feature_columns(prepared_factory):
    manifest, _, _ = prepared_factory(dimension=7, teacher=None)
    row = load_manifest(manifest).train[0]
    table = pq.read_table(row.embedding_file)
    metadata = {**table.schema.metadata, b"embedding_dimension": b"6"}
    pq.write_table(table.replace_schema_metadata(metadata), row.embedding_file)
    with pytest.raises(ValueError, match="embedding metadata"):
        load_embeddings(row)


@pytest.mark.parametrize("column", ["residue_id", "amino_acid", "record_id"])
def test_parquet_checks_residue_mapping(prepared_factory, column):
    manifest, _, _ = prepared_factory()
    row = load_manifest(manifest).train[0]
    table = pq.read_table(row.embedding_file)
    original = table[column]
    values = original.to_pylist()
    if column in {"residue_index", "residue_id"}:
        values[0], values[1] = values[1], values[0]
    else:
        values[0] = "wrong"
    table = table.set_column(table.column_names.index(column), column, pa.array(values, type=original.type))
    pq.write_table(table, row.embedding_file)
    with pytest.raises(ValueError, match=column):
        load_embeddings(row)


@pytest.mark.parametrize("kind,match", [
    ("nan", "float32"), ("float64", "types"), ("shape", "row count"),
])
def test_invalid_target_values(prepared_factory, kind, match):
    manifest, _, arrays = prepared_factory()
    row = load_manifest(manifest).train[0]
    values = arrays[0].copy()
    if kind == "nan":
        values[0, 0] = np.nan
    elif kind == "float64":
        values = values.astype(np.float64)
    table = pq.read_table(row.embedding_file)
    if kind == "shape":
        table = table.slice(0, len(values) - 1)
    else:
        names = table.column_names[:9]
        table = pa.table({
            **{name: table[name] for name in names},
            **{name: pa.array(values[:, i]) for i, name in enumerate(row.embedding_spec.feature_columns)},
        }).replace_schema_metadata(table.schema.metadata)
    pq.write_table(table, row.embedding_file)
    with pytest.raises(ValueError, match=match):
        load_embeddings(row)


def test_hub_snapshot_records_resolved_commit_and_keeps_relative_paths(
    prepared_factory, tmp_path,
):
    manifest, _, _ = prepared_factory()
    blob = tmp_path / "manifest-blob"
    manifest.rename(blob)
    manifest.symlink_to(blob)
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.parent.mkdir()
    manifest.parent.rename(snapshot)
    config = DatasetConfig(path="org/data", revision="main")
    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
        prepared = load_dataset(config)
    assert prepared.manifest == snapshot / "dataset.json"
    assert prepared.hub_revision == "a" * 40
    assert load_embeddings(prepared.train[0]).shape == (64, 7)
    download.assert_called_once_with(repo_id="org/data", repo_type="dataset", revision="main")


def test_local_dataset_directory_does_not_contact_hub(prepared_factory):
    manifest, _, _ = prepared_factory()
    with patch("huggingface_hub.snapshot_download") as download:
        prepared = load_dataset(DatasetConfig(path=str(manifest.parent)))
    download.assert_not_called()
    assert prepared.manifest == manifest
    assert prepared.hub_revision is None
    assert load_embeddings(prepared.train[0]).shape == (64, 7)


def test_missing_local_directory_never_falls_back_to_hub(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("huggingface_hub.snapshot_download") as download:
        with pytest.raises(FileNotFoundError, match="Dataset directory"):
            load_dataset(DatasetConfig(path="./missing"))
    download.assert_not_called()
