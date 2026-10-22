"""Tests for source-specific label construction in the sourcing scripts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl

from modules.evaluate.scripts.dataset_sourcing.source_uniprot_annotations import (
    AMBIGUOUS_LABEL,
    BACKGROUND_LABEL,
    _frame_from_uniprot_table,
    build_residue_labels,
    merge_residue_labels,
)
from modules.evaluate.scripts.dataset_sourcing.source_uniprot_go import (
    frame_from_uniprot_table,
    propagate_go_ids,
    read_go_ontology,
)
from modules.evaluate.src.dataset_sourcing import hf_io, pipeline
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec
_EVIDENCE = '/evidence="ECO:0000269|PubMed:1"'


def test_disulfide_bond_labels_only_its_two_cysteines():
    labels, _ = build_residue_labels(
        {"Disulfide bond": f"DISULFID 2..5; {_EVIDENCE}"}, sequence_length=6
    )

    assert labels == ["-", "DISULFID", "-", "-", "DISULFID", "-"]


def test_evidence_rejected_feature_residues_are_ambiguous_not_background():
    labels, rejected = build_residue_labels(
        {
            "Helix": (
                f"HELIX 1..2; {_EVIDENCE}; "
                'HELIX 4..5; /evidence="ECO:0000250|UniProtKB:P1"'
            )
        },
        sequence_length=6,
    )

    assert rejected
    assert labels == [
        "HELIX",
        "HELIX",
        BACKGROUND_LABEL,
        AMBIGUOUS_LABEL,
        AMBIGUOUS_LABEL,
        BACKGROUND_LABEL,
    ]


def test_uncertain_feature_positions_are_ambiguous_and_flagged():
    labels, rejected = build_residue_labels(
        {"Helix": f"HELIX ?2..3; {_EVIDENCE}; HELIX ?..5; {_EVIDENCE}"},
        sequence_length=4,
    )

    assert rejected
    assert labels == [
        BACKGROUND_LABEL,
        AMBIGUOUS_LABEL,
        AMBIGUOUS_LABEL,
        BACKGROUND_LABEL,
    ]


def test_merge_residue_labels_keeps_complementary_and_masks_conflicts():
    assert merge_residue_labels(["-", "A", "A", "?"], ["B", "-", "B", "-"]) == [
        "B",
        "A",
        AMBIGUOUS_LABEL,
        AMBIGUOUS_LABEL,
    ]


def test_duplicate_sequences_merge_evidence_flags():
    raw = pl.DataFrame(
        {
            "Entry": ["P1", "P2"],
            "Sequence": ["MKTA", "MKTA"],
            "Helix": [None, 'HELIX 1..2; /evidence="ECO:0000250"'],
            "Beta strand": [None, None],
            "Turn": [None, None],
        }
    )

    frame, _ = _frame_from_uniprot_table(raw, "uniprot_secondary_structure")

    assert frame["id"].to_list() == ["P1"]
    assert frame["_has_evidence_rejected_feature"].to_list() == [True]


def test_glycosylation_note_drops_parenthetical_text():
    labels, _ = build_residue_labels(
        {
            "Glycosylation": (
                f'CARBOHYD 2; /note="N-linked (GlcNAc...) asparagine"; {_EVIDENCE}'
            )
        },
        sequence_length=3,
        use_note=True,
    )

    assert labels[1] == "N-linked_asparagine"


def _write_obo(path: Path) -> Path:
    path.write_text(
        "format-version: 1.2\n"
        "data-version: releases/2026-01-01\n\n"
        "[Term]\nid: GO:0003674\nname: molecular_function\n\n"
        "[Term]\nid: GO:0000001\nname: parent\nis_a: GO:0003674 ! molecular_function\n\n"
        "[Term]\nid: GO:0000002\nname: child\nalt_id: GO:0000009\n"
        "is_a: GO:0000001 ! parent\n\n"
        "[Typedef]\nid: part_of\nname: part of\n",
        encoding="utf-8",
    )
    return path


def test_go_annotations_propagate_to_ancestors_without_root(tmp_path):
    ontology = read_go_ontology(_write_obo(tmp_path / "go.obo"))

    assert ontology.version == "releases/2026-01-01"
    assert propagate_go_ids({"GO:0000009"}, ontology) == {"GO:0000001", "GO:0000002"}

    raw = pl.DataFrame(
        {
            "Entry": ["P1", "P2"],
            "Sequence": ["MKT", "AAA"],
            "Gene Ontology (molecular function)": [
                "child [GO:0000002]",
                "parent [GO:0000001]",
            ],
        }
    )
    frame, vocabulary = frame_from_uniprot_table(
        raw, "molecular_function", min_label_count=2, ontology=ontology
    )

    assert vocabulary == ["GO:0000001"]
    assert frame["id"].to_list() == ["P1", "P2"]
    assert frame["targets"].to_list() == [[1], [1]]


def test_pipeline_reprocesses_when_config_changes(tmp_path):
    data_dir = tmp_path / "download" / "data"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"seq": ["AAA", "CCC"], "label": [0, 1]}).write_parquet(
        data_dir / "train-0.parquet"
    )
    pl.DataFrame({"seq": ["GGG"], "label": [0]}).write_parquet(
        data_dir / "valid-0.parquet"
    )
    pl.DataFrame({"seq": ["TTT"], "label": [1]}).write_parquet(
        data_dir / "test-0.parquet"
    )
    spec = DatasetSourceSpec(
        known_datasets=["example"],
        repo_id_template="source/{dataset_name}",
        column_rename={"seq": "sequence", "label": "targets"},
    )

    def fake_stats(output_dir, split_frames, threads, seqkit_split_frames=None):
        (output_dir / "stats.json").write_text(
            '{"rows_total": 0, "rows_by_split": {}, "columns": [], "seqkit_stats": []}',
            encoding="utf-8",
        )

    def run(config: DatasetSourcingConfig) -> None:
        pipeline.run_for_dataset(
            "example", config, None, tmp_path / "downloads", tmp_path / "out", spec
        )

    with (
        patch.object(pipeline, "download_dataset", return_value=data_dir),
        patch.object(pipeline, "write_dataset_stats", side_effect=fake_stats),
        patch.object(pipeline, "load_splits", wraps=pipeline.load_splits) as loader,
    ):
        run(DatasetSourcingConfig(datasets=["example"], seed=1))
        run(DatasetSourcingConfig(datasets=["example"], seed=1))
        run(DatasetSourcingConfig(datasets=["example"], seed=2))
        # Selection, threads, and commit message do not change the generated files.
        run(
            DatasetSourcingConfig(
                datasets=["example", "other"],
                seed=2,
                stats_threads=1,
                commit_message="unrelated",
            )
        )
        pl.DataFrame({"seq": ["TTT", "WWW"], "label": [1, 0]}).write_parquet(
            data_dir / "test-0.parquet"
        )
        run(DatasetSourcingConfig(datasets=["example"], seed=2))

    assert loader.call_count == 3


def test_upload_is_skipped_without_a_target(tmp_path):
    config = DatasetSourcingConfig(datasets=["example"])

    with patch.object(hf_io, "upload_to_hf") as upload:
        hf_io.upload_dataset_artifact(tmp_path, "example", config, token=None)

    upload.assert_not_called()


def test_upload_deletes_stale_remote_artifacts(tmp_path):
    config = DatasetSourcingConfig(datasets=["example"], repo_owner="owner")

    with patch.object(hf_io, "upload_to_hf") as upload:
        hf_io.upload_dataset_artifact(tmp_path, "example", config, token=None)

    kwargs = upload.call_args.kwargs
    assert kwargs["delete_patterns"] == kwargs["allow_patterns"]
    assert "data/**" in kwargs["delete_patterns"]
