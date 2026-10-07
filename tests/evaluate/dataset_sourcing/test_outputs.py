from __future__ import annotations

import json
from unittest.mock import patch

import polars as pl

from modules.evaluate.src.dataset_sourcing.outputs import (
    stats_json_is_complete,
    write_dataset_stats,
    write_seqkit_stats,
    write_stats,
)


def test_seqkit_stats_are_in_stats_json_and_legacy_parquet_is_removed(tmp_path):
    split_frames = {
        "train": pl.DataFrame({"id": ["p1"], "sequence": ["MKT"]})
    }
    write_stats(tmp_path, split_frames)
    stats_path = tmp_path / "stats.json"
    assert not stats_json_is_complete(stats_path)
    legacy_stats_dir = tmp_path / "stats"
    legacy_stats_dir.mkdir()
    legacy_stats_path = legacy_stats_dir / "example_stats.parquet"
    legacy_stats_path.touch()
    seqkit_stats = pl.DataFrame(
        {
            "file": ["train.fasta.gz", "overall"],
            "num_seqs": [1, 1],
            "sum_len": [3, 3],
            "avg_len": [3.0, 3.0],
        }
    )

    with patch(
        "modules.evaluate.src.dataset_sourcing.outputs.compute_seqkit_stats",
        return_value=seqkit_stats,
    ):
        stats_path = write_seqkit_stats(
            output_dir=tmp_path,
            dataset_name="example",
            split_frames=split_frames,
            threads=1,
        )

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats_json_is_complete(stats_path)
    assert stats["rows_total"] == 1
    assert stats["rows_by_split"] == {"train": 1}
    assert stats["columns"] == ["id", "sequence"]
    assert stats["seqkit_stats"] == seqkit_stats.to_dicts()
    assert not legacy_stats_path.exists()
    assert not legacy_stats_dir.exists()


def test_write_dataset_stats_can_use_distinct_seqkit_split_frames(tmp_path):
    stats_frames = {"train": pl.DataFrame({"id": ["variant-1"]})}
    seqkit_frames = {
        "all": pl.DataFrame({"id": ["variant-1"], "sequence": ["MKT"]})
    }

    with (
        patch(
            "modules.evaluate.src.dataset_sourcing.outputs.write_stats"
        ) as stats_writer,
        patch(
            "modules.evaluate.src.dataset_sourcing.outputs.write_seqkit_stats",
            return_value=tmp_path / "stats.json",
        ) as seqkit_writer,
    ):
        result = write_dataset_stats(
            tmp_path,
            "example",
            stats_frames,
            threads=2,
            seqkit_split_frames=seqkit_frames,
        )

    stats_writer.assert_called_once_with(tmp_path, stats_frames)
    seqkit_writer.assert_called_once_with(
        output_dir=tmp_path,
        dataset_name="example",
        split_frames=seqkit_frames,
        threads=2,
    )
    assert result == tmp_path / "stats.json"
