from __future__ import annotations

import gzip
import json
from unittest.mock import patch

import polars as pl
import pytest

from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.outputs import (
    build_seqkit_split_fastas,
    stats_json_is_complete,
    write_dataset_card,
    write_dataset_stats,
)
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec
from modules.evaluate.src.utils.stats import compute_seqkit_stats


@pytest.mark.parametrize("suffix", [".fasta", ".fasta.gz"])
@pytest.mark.parametrize("shard_count", [1, 2])
@pytest.mark.parametrize("combined_exists", [False, True])
def test_compute_seqkit_stats_preserves_shards_and_overall(
    tmp_path, suffix, shard_count, combined_exists
):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shards = [shard_dir / f"split_{index}{suffix}" for index in range(shard_count)]
    for shard in shards:
        shard.touch()
    combined = tmp_path / f"combined{suffix}"
    if combined_exists:
        combined.touch()

    def run_seqkit(command, output_path, input_files=None):
        assert command[:2] == ["seqkit", "stats"]
        assert "-a" in command and "-T" in command
        assert command[command.index("-j") + 1] == "1"
        if input_files is not None:
            assert input_files == shards
            assert command[-1] == "-"
            assert "-b" not in command
            files = ["-"]
        else:
            assert "-b" in command
            expected = shards + (
                [combined] if combined_exists and shard_count > 1 else []
            )
            assert command[command.index("-j") + 2 :] == [
                str(path) for path in expected
            ]
            files = [path.name for path in expected]
        pl.DataFrame(
            {"file": files, "num_seqs": [1] * len(files), "extra": [0] * len(files)}
        ).write_csv(output_path, separator="\t")

    with patch(
        "modules.evaluate.src.utils.stats._run_seqkit", side_effect=run_seqkit
    ) as runner:
        frame = compute_seqkit_stats(shard_dir, combined, threads=1)

    expected_files = [path.name for path in shards]
    if shard_count > 1:
        expected_files.append("overall")
    assert frame["file"].to_list() == expected_files
    assert frame.columns == ["file", "num_seqs"]
    assert runner.call_count == (2 if shard_count > 1 and not combined_exists else 1)
    assert sorted(shard_dir.iterdir()) == shards


def test_compute_seqkit_stats_empty_directory_does_not_run_seqkit(tmp_path):
    with patch("modules.evaluate.src.utils.stats._run_seqkit") as runner:
        assert compute_seqkit_stats(tmp_path, tmp_path / "combined.fasta").is_empty()
    runner.assert_not_called()


def _write_stats(tmp_path, split_frames, **kwargs):
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
    ) as compute:
        stats_path = write_dataset_stats(tmp_path, split_frames, threads=1, **kwargs)
    return stats_path, json.loads(stats_path.read_text(encoding="utf-8")), compute


def test_stats_json_combines_row_counts_and_seqkit_metrics(tmp_path):
    split_frames = {"train": pl.DataFrame({"id": ["p1"], "sequence": ["MKT"]})}

    stats_path, stats, _ = _write_stats(tmp_path, split_frames)

    assert stats_json_is_complete(stats_path)
    assert stats["rows_total"] == 1
    assert stats["rows_by_split"] == {"train": 1}
    assert stats["columns"] == ["id", "sequence"]
    assert stats["seqkit_stats"][-1]["file"] == "overall"


def test_stats_json_without_seqkit_metrics_is_incomplete(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"rows_total": 1, "rows_by_split": {}, "columns": []}', encoding="utf-8"
    )

    assert not stats_json_is_complete(stats_path)


def test_stats_count_canonical_rows_once(tmp_path):
    frame = pl.DataFrame({"id": ["p1"], "sequence": ["MKT"]})
    split_frames = {
        name: frame
        for name in ("train", "validation", "test", "train_random", "test_random")
    }

    _, stats, _ = _write_stats(tmp_path, split_frames)

    assert stats["rows_total"] == 5
    assert stats["canonical_rows"] == 3


def test_write_dataset_stats_can_use_distinct_seqkit_split_frames(tmp_path):
    stats_frames = {"train": pl.DataFrame({"id": ["variant-1"]})}
    seqkit_frames = {"all": pl.DataFrame({"id": ["variant-1"], "sequence": ["MKT"]})}

    _, stats, compute = _write_stats(
        tmp_path, stats_frames, seqkit_split_frames=seqkit_frames
    )

    assert stats["columns"] == ["id"]
    with gzip.open(compute.call_args.kwargs["fasta_output"], "rt") as handle:
        assert handle.read() == ">variant-1\nMKT\n"


def test_combined_seqkit_fasta_counts_only_canonical_splits(tmp_path):
    frame = pl.DataFrame({"id": ["p1"], "sequence": ["MKT"]})
    split_frames = {"train": frame, "train_random": frame, "train_stratified": frame}

    split_dir, combined = build_seqkit_split_fastas(tmp_path, split_frames)

    with gzip.open(combined, "rt", encoding="utf-8") as handle:
        assert handle.read() == ">p1\nMKT\n"
    assert len(list(split_dir.iterdir())) == 3


def test_dataset_card_documents_sequence_deduplication(tmp_path):
    card_path = write_dataset_card(
        output_dir=tmp_path,
        dataset_name="example",
        config=DatasetSourcingConfig(datasets=["example"]),
        split_frames={
            "train": pl.DataFrame(
                {
                    "id": ["example_train_0"],
                    "sequence": ["MKT"],
                    "targets": [0],
                    "split": ["train"],
                }
            )
        },
        validation_method="source",
        spec=DatasetSourceSpec(
            known_datasets=["example"], repo_id_template="source/{dataset_name}"
        ),
    )

    card = card_path.read_text(encoding="utf-8")
    assert "Duplicate sequences were removed within and across canonical" in card
    assert "kept in `test` first" in card
