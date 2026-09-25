"""Unit tests for the data assembly step."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from modules.data.src.steps.assemble import AssembleConfig, assemble_data


def test_assemble_config_has_expected_defaults() -> None:
    """The assembly step is enabled with the standard identifier columns."""
    config = AssembleConfig()

    assert config.enabled is True
    assert config.base_id_column == "sequence_id"
    assert config.cluster_id_column == "sequence_id"
    assert config.score_id_column == "sequence_id"
    assert config.skip_existing is True


def test_assemble_data_joins_shards(tmp_path: Path) -> None:
    """Assembly joins matching base and score shards with the cluster table."""
    base_path = tmp_path / "base"
    score_path = tmp_path / "scores"
    cluster_path = tmp_path / "clusters.parquet"
    output_path = tmp_path / "assembled"
    base_path.mkdir()
    score_path.mkdir()

    pl.DataFrame(
        {
            "sequence_id": ["seq1"],
            "sequence": ["ACDE"],
        }
    ).write_parquet(base_path / "shard_1.parquet")
    pl.DataFrame(
        {
            "sequence_id": ["seq2"],
            "sequence": ["KLM"],
        }
    ).write_parquet(base_path / "shard_2.parquet")
    pl.DataFrame(
        {
            "sequence_id": ["seq1", "seq2"],
            "cluster_id": ["cluster1", "cluster2"],
        }
    ).write_parquet(cluster_path)
    pl.DataFrame(
        {
            "sequence_id": ["seq1"],
            "score": [0.9],
        }
    ).write_parquet(score_path / "shard_1.parquet")
    pl.DataFrame(
        {
            "sequence_id": ["seq2"],
            "score": [0.8],
        }
    ).write_parquet(score_path / "shard_2.parquet")

    written = assemble_data(
        base_source=base_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        dataset_id="toy-dataset",
        cluster_source=cluster_path,
        score_source=score_path,
        max_workers=1,
        num_join_buckets=2,
    )

    assert written == [
        output_path / "shard_1.parquet",
        output_path / "shard_2.parquet",
    ]
    result = pl.concat(
        [
            pl.read_parquet(output_path / "shard_1.parquet"),
            pl.read_parquet(output_path / "shard_2.parquet"),
        ]
    ).sort("sequence_id")
    assert result.select(
        [
            "sequence_id",
            "sequence",
            "cluster_id",
            "dataset_id",
            "sequence_length",
            "score",
        ]
    ).to_dicts() == [
        {
            "sequence_id": "seq1",
            "sequence": "ACDE",
            "cluster_id": "cluster1",
            "dataset_id": "toy-dataset",
            "sequence_length": 4,
            "score": 0.9,
        },
        {
            "sequence_id": "seq2",
            "sequence": "KLM",
            "cluster_id": "cluster2",
            "dataset_id": "toy-dataset",
            "sequence_length": 3,
            "score": 0.8,
        },
    ]


def test_assemble_data_skips_existing_shards(tmp_path: Path) -> None:
    """Completed shard outputs are returned without being reprocessed."""
    base_path = tmp_path / "base"
    output_path = tmp_path / "assembled"
    base_path.mkdir()
    output_path.mkdir()
    shard_path = base_path / "shard_1.parquet"
    output_shard = output_path / shard_path.name
    pl.DataFrame({"sequence_id": ["seq1"], "sequence": ["ACDE"]}).write_parquet(
        shard_path
    )
    pl.DataFrame({"sequence_id": ["seq1"], "assembled": [True]}).write_parquet(
        output_shard
    )

    written = assemble_data(
        base_source=base_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        dataset_id="toy-dataset",
        cluster_source=tmp_path / "cluster-source-not-needed.parquet",
    )

    assert written == [output_shard]
    assert pl.read_parquet(output_shard)["assembled"].to_list() == [True]


def test_assemble_data_can_skip_scores(tmp_path: Path) -> None:
    """Assembly without a score source still writes the joined shard."""
    base_path = tmp_path / "base"
    output_path = tmp_path / "assembled"
    cluster_path = tmp_path / "clusters.parquet"
    base_path.mkdir()
    pl.DataFrame({"sequence_id": ["seq1"], "sequence": ["ACDE"]}).write_parquet(
        base_path / "shard_1.parquet"
    )
    pl.DataFrame({"sequence_id": ["seq1"], "cluster_id": ["cluster1"]}).write_parquet(
        cluster_path
    )

    assemble_data(
        base_source=base_path,
        output_path=output_path,
        work_dir=tmp_path / "work",
        dataset_id="toy-dataset",
        cluster_source=cluster_path,
        max_workers=1,
        num_join_buckets=2,
    )

    result = pl.read_parquet(output_path / "shard_1.parquet")
    assert result.columns == [
        "sequence_id",
        "sequence",
        "cluster_id",
        "dataset_id",
        "sequence_length",
    ]


def test_assemble_data_requires_parquet_shards(tmp_path: Path) -> None:
    """A missing or empty base directory fails clearly."""
    with pytest.raises(FileNotFoundError, match="No parquet files"):
        assemble_data(
            base_source=tmp_path / "base",
            output_path=tmp_path / "assembled",
            work_dir=tmp_path / "work",
            dataset_id="toy-dataset",
            cluster_source=tmp_path / "clusters.parquet",
        )
