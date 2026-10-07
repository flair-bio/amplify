"""Tests for modules.evaluate.src.dataset_sourcing.preprocess."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from modules.evaluate.src.dataset_sourcing import preprocess
from modules.evaluate.src.dataset_sourcing.preprocess import (
    _resolve_sequence_column,
    create_full_dataset_splits,
    create_split_subsets,
    create_validation_split_from_train,
    load_splits_with_method,
)


def _frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(n))})


def test_target_strata_rejects_mixed_scalar_and_token_targets():
    with pytest.raises(TypeError, match="mixture of scalar and token targets"):
        preprocess.target_strata([[0, 1], 0])


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

    def test_falls_back_to_random_sampling_when_mmseqs_missing(self):
        train_frame = pl.DataFrame(
            {"id": list(range(20)), "sequence": [f"SEQ{i}" for i in range(20)]}
        )

        with patch.object(preprocess.shutil, "which", return_value=None):
            train_out, validation_out, method = (
                preprocess._create_validation_split_from_train_with_method(
                train_frame=train_frame,
                test_frame=_frame(5),
                seed=0,
                sequence_column="sequence",
                )
            )

        assert method == "random"
        assert validation_out.height == 5
        assert train_out.height == 15
        assert set(train_out["id"]).isdisjoint(set(validation_out["id"]))

    def test_max_sequence_length_is_applied_before_synthesized_validation(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pl.DataFrame(
            {
                "sequence": ["AAAA", "CCC", "GGGG", "TT"],
                "targets": [0, 1, 0, 1],
            }
        ).write_parquet(data_dir / "train-000.parquet")
        pl.DataFrame({"sequence": ["AA"], "targets": [0]}).write_parquet(
            data_dir / "test-000.parquet"
        )

        with patch.object(preprocess.shutil, "which", return_value=None):
            split_frames, method = load_splits_with_method(
                data_dir, seed=0, max_sequence_length=3
            )

        assert method == "random"
        assert all(
            sequence_length <= 3
            for frame in split_frames.values()
            for sequence_length in frame["sequence"].str.len_chars().to_list()
        )

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
            train_out, validation_out, method = (
                preprocess._create_validation_split_from_train_with_method(
                train_frame=train_frame,
                test_frame=_frame(5),
                seed=0,
                sequence_column="sequence",
                )
            )

        assert method == "mmseqs2"
        assert set(train_out["id"]).isdisjoint(set(validation_out["id"]))
        # No cluster is split across train/validation.
        validation_ids = set(validation_out["id"])
        for row_idx, cluster in enumerate(cluster_of_row):
            cluster_row_ids = {i for i, c in enumerate(cluster_of_row) if c == cluster}
            assert cluster_row_ids <= validation_ids or cluster_row_ids.isdisjoint(
                validation_ids
            )

    def test_excludes_train_clusters_shared_with_upstream_test(self):
        train_frame = pl.DataFrame({
            "id": list(range(6)), "sequence": [f"SEQ{i}" for i in range(6)],
        })
        test_frame = pl.DataFrame({"id": [10], "sequence": ["SEQ0"]})
        assignments = pl.DataFrame({
            "sequence_id": [str(i) for i in range(6)] + ["test_0"],
            "cluster_id": ["shared", "shared", "a", "b", "c", "d", "shared"],
        })
        with (
            patch.object(preprocess.shutil, "which", return_value="/usr/bin/mmseqs"),
            patch.object(preprocess, "mmseqs_clustering", return_value=assignments),
        ):
            train_out, validation_out = create_validation_split_from_train(
                train_frame, test_frame, seed=0, sequence_column="sequence",
            )
        assert set(train_out["id"]) | set(validation_out["id"]) == {2, 3, 4, 5}


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

    def test_stratifies_regression_frames_by_a_numeric_column(self):
        lengths = [10] * 10 + [20] * 10
        all_rows = pl.DataFrame(
            {
                "id": [f"row-{index}" for index in range(20)],
                "sequence": [f"SEQUENCE{index}" for index in range(20)],
                "length": lengths,
            }
        )
        split_frames = {
            "train": all_rows.slice(0, 12),
            "validation": all_rows.slice(12, 4),
            "test": all_rows.slice(16, 4),
        }
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(20)],
                "cluster_id": [str(index) for index in range(20)],
            }
        )

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            result = create_split_subsets(
                split_frames, seed=7, stratification_column="length"
            )

        for split, expected_per_length in (
            ("train_stratified", 6),
            ("validation_stratified", 2),
            ("test_stratified", 2),
        ):
            counts = result[split].group_by("length").len().sort("length")
            assert counts["len"].to_list() == [expected_per_length] * 2

    def test_cluster_splits_are_skipped_when_clusters_cannot_populate_all_splits(
        self, caplog
    ):
        split_frames = {
            "train": _processed_frame(6),
            "validation": _processed_frame(3, offset=6),
            "test": _processed_frame(3, offset=9),
        }
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(12)],
                "cluster_id": ["cluster-a"] * 12,
            }
        )

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            result = create_split_subsets(split_frames, seed=7)

        assert "train_cluster" not in result
        assert "validation_cluster" not in result
        assert "test_cluster" not in result
        assert "train_random" in result
        assert "train_stratified" in result
        assert "Skipping optional cluster split subsets" in caplog.text

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


class TestCreateFullDatasetSplits:
    def test_canonical_cluster_splits_still_require_enough_clusters(self):
        frame = _processed_frame(10)
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(10)],
                "cluster_id": ["cluster-a"] * 10,
            }
        )

        with (
            patch.object(preprocess, "mmseqs_clustering", return_value=assignments),
            pytest.raises(ValueError, match="at least 3 clusters"),
        ):
            create_full_dataset_splits(
                frame,
                split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
                seed=7,
            )

    def test_rejects_fractions_not_summing_to_one(self):
        frame = _processed_frame(10)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_full_dataset_splits(
                frame, split_fractions={"train": 0.5, "test": 0.4}, seed=1
            )

    def test_partitions_every_row_along_cluster_boundaries(self):
        frame = _processed_frame(20)
        cluster_ids = [index // 2 for index in range(20)]
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(index) for index in range(20)],
                "cluster_id": [str(cluster_id) for cluster_id in cluster_ids],
            }
        )

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            result = create_full_dataset_splits(
                frame,
                split_fractions={"train": 0.8, "validation": 0.1, "test": 0.1},
                seed=7,
            )

        assert set(result) == {"train", "validation", "test"}
        assert sum(frame.height for frame in result.values()) == 20
        all_ids = {id_ for frame in result.values() for id_ in frame["id"].to_list()}
        assert all_ids == set(frame["id"].to_list())

        # No cluster (pair of consecutive rows) is split across splits.
        id_to_split = {
            row_id: split_name
            for split_name, split_frame in result.items()
            for row_id in split_frame["id"]
        }
        for cluster_id in set(cluster_ids):
            members = [f"row-{i}" for i, c in enumerate(cluster_ids) if c == cluster_id]
            assert len({id_to_split[member] for member in members}) == 1


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
