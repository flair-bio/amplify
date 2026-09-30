"""Tests for modules.evaluate.src.dataset_sourcing.preprocess."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.scripts.dataset_sourcing.source_biomap_datasets import (
    SourceBiomapDatasetsConfig,
)
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing import preprocess
from modules.evaluate.src.dataset_sourcing.preprocess import (
    _resolve_sequence_column,
    create_split_subsets,
    create_validation_split_from_train,
    minimal_preprocess,
)


def _frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(n))})


def test_split_subset_creation_is_yaml_controlled():
    assert DatasetSourcingConfig(datasets=["toy"]).create_split_subsets is False

    config = load_and_parse(
        "modules/evaluate/configs/dataset_sourcing/source_biomap_datasets.yaml",
        SourceBiomapDatasetsConfig,
    )

    assert config.create_split_subsets is True


class TestCreateValidationSplitFromTrain:
    def test_splits_are_disjoint_and_size_matched_to_test(self):
        train_frame, validation_frame = create_validation_split_from_train(
            train_frame=_frame(100), test_frame=_frame(10), seed=0
        )

        assert validation_frame.height == 10
        assert train_frame.height == 90
        assert set(train_frame["id"]).isdisjoint(set(validation_frame["id"]))

    def test_warns_when_validation_would_consume_large_share_of_train(self, caplog):
        with caplog.at_level("WARNING"):
            create_validation_split_from_train(
                train_frame=_frame(100), test_frame=_frame(40), seed=0
            )

        assert any("Synthesized validation split" in record.message for record in caplog.records)

    def test_no_warning_for_small_validation_share(self, caplog):
        with caplog.at_level("WARNING"):
            create_validation_split_from_train(
                train_frame=_frame(100), test_frame=_frame(10), seed=0
            )

        assert not any(
            "Synthesized validation split" in record.message for record in caplog.records
        )

    def test_falls_back_to_random_sampling_when_mmseqs_missing(self):
        train_frame = pl.DataFrame(
            {"id": list(range(20)), "sequence": [f"SEQ{i}" for i in range(20)]}
        )

        with patch.object(preprocess.shutil, "which", return_value=None):
            train_out, validation_out = create_validation_split_from_train(
                train_frame=train_frame,
                test_frame=_frame(5),
                seed=0,
                sequence_column="sequence",
            )

        assert validation_out.height == 5
        assert train_out.height == 15
        assert set(train_out["id"]).isdisjoint(set(validation_out["id"]))

    def test_clusters_stay_together_when_mmseqs_available(self):
        # 4 sequences per cluster, 5 clusters -> 20 rows total.
        cluster_of_row = [row_idx // 4 for row_idx in range(20)]
        train_frame = pl.DataFrame(
            {
                "id": list(range(20)),
                "sequence": [f"SEQ{i}" for i in range(20)],
            }
        )
        fake_assignments = pl.DataFrame(
            {
                "sequence_id": [str(i) for i in range(20)],
                "cluster_id": [str(c) for c in cluster_of_row],
            }
        )

        with (
            patch.object(preprocess.shutil, "which", return_value="/usr/bin/mmseqs"),
            patch.object(
                preprocess, "mmseqs_clustering", return_value=fake_assignments
            ),
        ):
            train_out, validation_out = create_validation_split_from_train(
                train_frame=train_frame,
                test_frame=_frame(5),
                seed=0,
                sequence_column="sequence",
            )

        assert set(train_out["id"]).isdisjoint(set(validation_out["id"]))
        # No cluster is split across train/validation.
        validation_ids = set(validation_out["id"])
        for row_idx, cluster in enumerate(cluster_of_row):
            cluster_row_ids = {i for i, c in enumerate(cluster_of_row) if c == cluster}
            assert cluster_row_ids <= validation_ids or cluster_row_ids.isdisjoint(
                validation_ids
            )


class TestResolveSequenceColumn:
    def test_returns_sequence_when_already_present(self):
        frame = pl.DataFrame({"sequence": ["A"]})

        assert _resolve_sequence_column(frame, None) == "sequence"

    def test_resolves_via_column_rename_mapping(self):
        frame = pl.DataFrame({"seq_raw": ["A"]})

        assert (
            _resolve_sequence_column(frame, {"seq_raw": "sequence"}) == "seq_raw"
        )

    def test_returns_none_when_unresolvable(self):
        frame = pl.DataFrame({"other": ["A"]})

        assert _resolve_sequence_column(frame, {"seq_raw": "sequence"}) is None


class TestCreateSplitSubsets:
    def test_preserves_defaults_and_creates_three_custom_split_families(self):
        split_frames = {
            "train": _processed_frame(12),
            "validation": _processed_frame(4, offset=12),
            "test": _processed_frame(4, offset=16),
        }
        cluster_ids = [index // 2 for index in range(20)]
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(20)],
                "cluster_id": [str(cluster_id) for cluster_id in cluster_ids],
            }
        )

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            result = create_split_subsets(split_frames, seed=7)

        assert set(result) == {
            "train",
            "validation",
            "test",
            "train_cluster",
            "validation_cluster",
            "test_cluster",
            "train_random",
            "validation_random",
            "test_random",
            "train_stratified",
            "validation_stratified",
            "test_stratified",
        }
        for role in ("train", "validation", "test"):
            assert result[role]["id"].to_list() == split_frames[role]["id"].to_list()

        for method in ("_cluster", "_random", "_stratified"):
            assert result[f"train{method}"].height == 12
            assert result[f"validation{method}"].height == 4
            assert result[f"test{method}"].height == 4

        canonical_split_by_id = {
            row_id: split
            for split in ("train_cluster", "validation_cluster", "test_cluster")
            for row_id in result[split]["id"]
        }
        for cluster_id in set(cluster_ids):
            rows = [f"row-{index}" for index, value in enumerate(cluster_ids) if value == cluster_id]
            assert len({canonical_split_by_id[row_id] for row_id in rows}) == 1

    def test_stratified_splits_preserve_binary_label_balance(self):
        split_frames = {
            "train": _processed_frame(12),
            "validation": _processed_frame(4, offset=12),
            "test": _processed_frame(4, offset=16),
        }
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(20)],
                "cluster_id": [str(index) for index in range(20)],
            }
        )

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            result = create_split_subsets(split_frames, seed=7)

        for split in ("train_stratified", "validation_stratified", "test_stratified"):
            counts = result[split].group_by("targets").len().sort("targets")
            assert counts["len"].to_list() == [result[split].height // 2] * 2

    def test_cluster_split_requires_one_cluster_per_output_split(self):
        split_frames = {
            "train": _processed_frame(6),
            "validation": _processed_frame(3, offset=6),
            "test": _processed_frame(3, offset=9),
        }
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(12)],
                "cluster_id": ["cluster-a"] * 6 + ["cluster-b"] * 6,
            }
        )

        with (
            patch.object(preprocess, "mmseqs_clustering", return_value=assignments),
            pytest.raises(ValueError, match="at least 3 clusters"),
        ):
            create_split_subsets(split_frames, seed=7)

    def test_cluster_split_is_independent_of_labels(self):
        split_frames = {
            "train": _processed_frame(12),
            "validation": _processed_frame(4, offset=12),
            "test": _processed_frame(4, offset=16),
        }
        cluster_ids = [index // 2 for index in range(20)]
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(20)],
                "cluster_id": [str(cluster_id) for cluster_id in cluster_ids],
            }
        )
        relabeled = {
            role: frame.with_columns((1 - pl.col("targets")).alias("targets"))
            for role, frame in split_frames.items()
        }

        with patch.object(
            preprocess, "mmseqs_clustering", return_value=assignments
        ):
            original = create_split_subsets(split_frames, seed=7)
            changed_labels = create_split_subsets(relabeled, seed=7)

        for role in ("train_cluster", "validation_cluster", "test_cluster"):
            assert original[role]["id"].to_list() == changed_labels[role]["id"].to_list()

def _processed_frame(n: int, offset: int = 0) -> pl.DataFrame:
    indices = list(range(offset, offset + n))
    return pl.DataFrame(
        {
            "id": [f"row-{index}" for index in indices],
            "sequence": [f"SEQ{index}" for index in indices],
            "targets": [index % 2 for index in indices],
            "split": ["source"] * n,
        }
    )
