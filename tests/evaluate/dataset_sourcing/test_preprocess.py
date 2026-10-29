"""Tests for modules.evaluate.scripts.dataset_sourcing.shared.preprocess."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

from modules.evaluate.scripts.dataset_sourcing.shared import preprocess
from modules.evaluate.scripts.dataset_sourcing.shared.config import (
    DatasetSourcingConfig,
    UniProtDatasetSourcingConfig,
)
from modules.evaluate.scripts.dataset_sourcing.shared.preprocess import (
    _resolve_renamed_column,
    clean_source_splits,
    create_full_dataset_splits,
    create_split_subsets,
    create_validation_split_from_train,
    filter_labels_by_split_support,
    load_splits,
    merge_label_sets_by_sequence,
    minimal_preprocess,
    read_source_splits,
)


def _frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({"id": list(range(n))})


def _config(seed: int = 0, **kwargs) -> DatasetSourcingConfig:
    return DatasetSourcingConfig(datasets=["example"], seed=seed, **kwargs)


def test_target_strata_rejects_mixed_scalar_and_token_targets():
    with pytest.raises(TypeError, match="mixture of scalar and token targets"):
        preprocess.target_strata([[0, 1], 0])


def test_target_strata_uses_rarest_positive_label_for_multilabel_targets():
    targets = [[1, 1, 0], [1, 0, 0], [1, 0, 1], [0, 0, 0]]

    assert preprocess.target_strata(targets, multilabel=True) == [
        "rarest:1",
        "rarest:0",
        "rarest:2",
        "rarest:none",
    ]


def test_merge_label_sets_by_sequence_unions_labels_and_keeps_first_id():
    merged = merge_label_sets_by_sequence(
        [
            ("P1", "AAAA", {"x"}),
            ("P2", "CCCC", {"y"}),
            ("P3", "AAAA", {"z"}),
        ]
    )

    assert merged == [("P1", "AAAA", {"x", "z"}), ("P2", "CCCC", {"y"})]


def test_filter_labels_by_split_support_uses_each_canonical_split():
    split_frames = {
        "train": pl.DataFrame({"targets": [[1, 1, 1], [1, 1, 0]]}),
        "validation": pl.DataFrame({"targets": [[1, 0, 1]]}),
        "test": pl.DataFrame({"targets": [[1, 1, 1]]}),
        "test_random": pl.DataFrame({"targets": [[0, 0, 1]]}),
    }

    filtered, retained = filter_labels_by_split_support(
        split_frames,
        label_count=3,
        min_support_by_split={"train": 2, "validation": 1, "test": 1},
    )

    assert retained == [0]
    assert filtered["train"]["targets"].to_list() == [[1], [1]]
    assert filtered["validation"]["targets"].to_list() == [[1]]
    assert filtered["test"]["targets"].to_list() == [[1]]
    assert filtered["test_random"]["targets"].to_list() == [[0]]


def test_filter_labels_by_split_support_drops_rows_left_without_labels():
    split_frames = {
        "train": pl.DataFrame({"id": [1, 2], "targets": [[1, 0], [0, 1]]}),
        "validation": pl.DataFrame({"id": [3], "targets": [[1, 0]]}),
        "test": pl.DataFrame({"id": [4], "targets": [[1, 1]]}),
    }

    filtered, retained = filter_labels_by_split_support(
        split_frames,
        label_count=2,
        min_support_by_split={"train": 1, "validation": 1, "test": 1},
    )

    assert retained == [0]
    assert filtered["train"]["id"].to_list() == [1]
    assert filtered["train"]["targets"].dtype == pl.List(pl.Int8)


def test_clean_source_splits_dedupes_with_eval_splits_taking_priority():
    split_frames = {
        "train": pl.DataFrame(
            {
                "seq": [" AAA ", "AAA", "CCC", None, "GGG", "TTTT"],
                "label": [0, 1, 2, 3, None, 4],
            }
        ),
        "test": pl.DataFrame({"seq": ["CCC", "AA"], "label": [5, 6]}),
    }

    result = clean_source_splits(
        split_frames,
        max_sequence_length=3,
        column_rename={"seq": "sequence", "label": "targets"},
    )

    assert list(result) == ["train", "test"]
    assert result["train"]["seq"].to_list() == ["AAA"]
    assert result["test"]["seq"].to_list() == ["CCC", "AA"]


def test_minimal_preprocess_keeps_existing_ids_and_generates_missing_ones():
    split_frames = {
        "train": pl.DataFrame({"id": ["P1"], "sequence": ["MKT"], "targets": [0]}),
        "val": pl.DataFrame({"seq": ["AAA"], "label": [1]}),
    }

    result = minimal_preprocess(
        dataset_name="example",
        split_frames=split_frames,
        column_rename={"seq": "sequence", "label": "targets"},
    )

    assert result["train"]["id"].to_list() == ["P1"]
    assert result["validation"]["id"].to_list() == ["example_validation_0"]
    assert result["validation"].columns == ["id", "sequence", "targets", "split"]


def test_read_source_splits_does_not_synthesize_validation(tmp_path):
    pl.DataFrame({"seq": ["AAA", "CCC"]}).write_parquet(tmp_path / "train-0.parquet")
    pl.DataFrame({"seq": ["GGG"]}).write_parquet(tmp_path / "test-0.parquet")

    assert set(read_source_splits(tmp_path)) == {"train", "test"}


def test_read_source_splits_reads_every_shard(tmp_path):
    pl.DataFrame({"seq": ["AAA"]}).write_parquet(
        tmp_path / "train-00000-of-00002.parquet"
    )
    pl.DataFrame({"seq": ["CCC"]}).write_parquet(
        tmp_path / "train-00001-of-00002.parquet"
    )
    pl.DataFrame({"seq": ["GGG"]}).write_parquet(
        tmp_path / "test-00000-of-00001.parquet"
    )

    assert read_source_splits(tmp_path)["train"]["seq"].to_list() == ["AAA", "CCC"]


class TestCreateValidationSplitFromTrain:
    def test_splits_are_disjoint_and_size_matched_to_test(self):
        train_frame, validation_frame, method = create_validation_split_from_train(
            train_frame=_frame(100), test_frame=_frame(10), config=_config()
        )

        assert method == "random"
        assert validation_frame.height == 10
        assert train_frame.height == 90
        assert set(train_frame["id"]).isdisjoint(set(validation_frame["id"]))

    def test_warns_when_validation_would_consume_large_share_of_train(self, caplog):
        with caplog.at_level("WARNING"):
            create_validation_split_from_train(
                train_frame=_frame(100), test_frame=_frame(40), config=_config()
            )

        assert any(
            "Synthesized validation split" in record.message
            for record in caplog.records
        )

    def test_falls_back_to_random_sampling_when_mmseqs_missing(self, caplog):
        train_frame = pl.DataFrame(
            {"id": list(range(20)), "sequence": [f"SEQ{i}" for i in range(20)]}
        )

        with patch.object(preprocess.shutil, "which", return_value=None):
            train_out, validation_out, method = create_validation_split_from_train(
                train_frame=train_frame,
                test_frame=_frame(5),
                config=_config(),
                sequence_column="sequence",
            )

        assert method == "random"
        assert validation_out.height == 5
        assert train_out.height == 15
        assert set(train_out["id"]).isdisjoint(set(validation_out["id"]))
        assert any(
            record.levelname == "WARNING" and "mmseqs not found" in record.message
            for record in caplog.records
        )

    def test_max_sequence_length_is_applied_before_synthesized_validation(
        self, tmp_path
    ):
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
            split_frames, method = load_splits(data_dir, _config(max_sequence_length=3))

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
            train_out, validation_out, method = create_validation_split_from_train(
                train_frame=train_frame,
                test_frame=_frame(5),
                config=_config(),
                sequence_column="sequence",
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

    def test_validation_selection_ignores_mmseqs_output_order(self):
        train_frame = pl.DataFrame(
            {"id": list(range(20)), "sequence": [f"SEQ{i}" for i in range(20)]}
        )
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(i) for i in range(20)],
                "cluster_id": [f"c{i // 2}" for i in range(20)],
            }
        )

        def validation_ids(order: pl.DataFrame) -> list[int]:
            with (
                patch.object(
                    preprocess.shutil, "which", return_value="/usr/bin/mmseqs"
                ),
                patch.object(preprocess, "mmseqs_clustering", return_value=order),
            ):
                _, validation, _ = create_validation_split_from_train(
                    train_frame, _frame(4), _config(seed=3), sequence_column="sequence"
                )
            return validation["id"].to_list()

        assert validation_ids(assignments) == validation_ids(assignments.reverse())

    def test_keeps_test_homologs_in_train_but_out_of_validation(self):
        train_frame = pl.DataFrame(
            {
                "id": list(range(6)),
                "sequence": [f"SEQ{i}" for i in range(6)],
            }
        )
        test_frame = pl.DataFrame({"id": [10], "sequence": ["SEQ0"]})
        assignments = pl.DataFrame(
            {
                "sequence_id": [str(i) for i in range(6)] + ["test_0"],
                "cluster_id": ["shared", "shared", "a", "b", "c", "d", "shared"],
            }
        )
        with (
            patch.object(preprocess.shutil, "which", return_value="/usr/bin/mmseqs"),
            patch.object(preprocess, "mmseqs_clustering", return_value=assignments),
        ):
            train_out, validation_out, _ = create_validation_split_from_train(
                train_frame,
                test_frame,
                _config(),
                sequence_column="sequence",
            )
        assert set(train_out["id"]) | set(validation_out["id"]) == set(range(6))
        assert {0, 1} <= set(train_out["id"])
        assert validation_out.height == 1


class TestResolveRenamedColumn:
    def test_returns_name_when_already_present(self):
        frame = pl.DataFrame({"sequence": ["A"]})

        assert _resolve_renamed_column(frame, None, "sequence") == "sequence"

    def test_resolves_via_column_rename_mapping(self):
        frame = pl.DataFrame({"seq_raw": ["A"]})

        assert (
            _resolve_renamed_column(frame, {"seq_raw": "sequence"}, "sequence")
            == "seq_raw"
        )

    def test_returns_none_when_unresolvable(self):
        frame = pl.DataFrame({"other": ["A"]})

        assert (
            _resolve_renamed_column(frame, {"seq_raw": "sequence"}, "sequence") is None
        )


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
            result = create_split_subsets(split_frames, 7, _config(seed=7))

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
            rows = [
                f"row-{index}"
                for index, value in enumerate(cluster_ids)
                if value == cluster_id
            ]
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
            result = create_split_subsets(split_frames, 7, _config(seed=7))

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
                split_frames, 7, _config(seed=7), stratification_column="length"
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
            result = create_split_subsets(split_frames, 7, _config(seed=7))

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

        with patch.object(preprocess, "mmseqs_clustering", return_value=assignments):
            original = create_split_subsets(split_frames, 7, _config(seed=7))
            changed_labels = create_split_subsets(relabeled, 7, _config(seed=7))

        for role in ("train_cluster", "validation_cluster", "test_cluster"):
            assert (
                original[role]["id"].to_list() == changed_labels[role]["id"].to_list()
            )

    def test_skips_stratified_subsets_for_contact_pair_targets(self, caplog):
        split_frames = {
            role: _processed_frame(4, offset=offset).with_columns(
                pl.Series("targets", [[[0, 0], [0, 1]]] * 4)
            )
            for role, offset in (("train", 0), ("validation", 4), ("test", 8))
        }

        result = create_split_subsets(split_frames, seed=7)

        assert "train_random" in result
        assert "train_stratified" not in result
        assert "Skipping stratified split subsets" in caplog.text

    def test_skips_stratified_subsets_when_targets_form_one_stratum(self, caplog):
        split_frames = {
            role: _processed_frame(4, offset=offset).with_columns(
                pl.lit(1).alias("targets")
            )
            for role, offset in (("train", 0), ("validation", 4), ("test", 8))
        }

        result = create_split_subsets(split_frames, seed=7)

        assert "train_stratified" not in result
        assert "single stratum" in caplog.text


def _uniprot_config(
    seed: int, **split_fractions: float
) -> UniProtDatasetSourcingConfig:
    return UniProtDatasetSourcingConfig(
        datasets=["example"],
        seed=seed,
        split_fractions=split_fractions
        or {"train": 0.8, "validation": 0.1, "test": 0.1},
    )


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
            create_full_dataset_splits(frame, _uniprot_config(seed=7))

    def test_rejects_fractions_not_summing_to_one(self):
        frame = _processed_frame(10)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_full_dataset_splits(
                frame, _uniprot_config(seed=1, train=0.5, test=0.4)
            )

    def test_rejects_frames_too_small_for_every_split(self):
        frame = _processed_frame(4)
        with pytest.raises(ValueError, match="too few"):
            create_full_dataset_splits(frame, _uniprot_config(seed=1))

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
            result = create_full_dataset_splits(frame, _uniprot_config(seed=7))

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
