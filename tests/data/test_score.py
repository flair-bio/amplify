"""Unit tests for shard balancing/assignment in the score step."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from modules.data.src.steps.score import ScoreConfig, ScoreStep, _ScoreCollator


def _write_shard(path: Path, num_rows: int) -> Path:
    table = pa.table({"sequence_id": [str(i) for i in range(num_rows)]})
    pq.write_table(table, path)
    return path


def _make_shards(tmp_path: Path, row_counts: list[int]) -> list[Path]:
    return [
        _write_shard(tmp_path / f"shard_{i}.parquet", rows)
        for i, rows in enumerate(row_counts)
    ]


def test_balance_shards_isolates_a_dominant_large_shard(tmp_path: Path):
    # One huge shard plus several tiny ones: naive round-robin (i % count)
    # would place shard 0 and shard 2 (huge + a tiny one) in the same
    # bucket, doubling that bucket's runtime versus the other. LPT
    # balancing should instead give the huge shard its own bucket and pack
    # all the tiny ones into the other, which is the best achievable split.
    shards = _make_shards(tmp_path, [1000, 10, 10, 10, 10])

    buckets = ScoreStep._balance_shards(shards, count=2)

    loads = [sum(pq.ParquetFile(p).metadata.num_rows for p in bucket) for bucket in buckets]
    assert sum(loads) == 1040
    assert sorted(loads) == [40, 1000]
    heavy_bucket = next(b for b in buckets if len(b) == 1)
    assert pq.ParquetFile(heavy_bucket[0]).metadata.num_rows == 1000


def test_balance_shards_preserves_all_shards_exactly_once(tmp_path: Path):
    shards = _make_shards(tmp_path, [50, 5, 30, 8, 12, 1])

    buckets = ScoreStep._balance_shards(shards, count=3)

    flattened = [p for bucket in buckets for p in bucket]
    assert sorted(flattened) == sorted(shards)


def test_balance_shards_handles_more_buckets_than_shards(tmp_path: Path):
    shards = _make_shards(tmp_path, [10, 20])

    buckets = ScoreStep._balance_shards(shards, count=5)

    assert len(buckets) == 5
    assert sum(len(bucket) for bucket in buckets) == len(shards)


def test_get_assigned_shards_uses_slurm_array_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    shards = _make_shards(tmp_path, [1000, 10, 10, 10, 10])
    output_dir = tmp_path / "scores"
    output_dir.mkdir()
    step = ScoreStep(ScoreConfig())

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "0")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", "1")

    assigned = step._get_assigned_shards(shards, output_dir)

    # Task 1 (index 1 of 2) should get the small shards balanced against
    # task 0, not an arbitrary round-robin slice.
    assigned_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in assigned)
    assert assigned_rows in (1000, 40)


def test_get_assigned_shards_respects_explicit_overrides(tmp_path: Path):
    shards = _make_shards(tmp_path, [1000, 10, 10, 10, 10])
    output_dir = tmp_path / "scores"
    output_dir.mkdir()
    step = ScoreStep(ScoreConfig(shard_index=0, shard_count=2))

    assigned = step._get_assigned_shards(shards, output_dir)

    assigned_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in assigned)
    assert assigned_rows == 1000


def test_get_assigned_shards_resumes_by_rebalancing_pending_shards(tmp_path: Path):
    # Simulate a crashed run resubmitted with a smaller array: shard_0 already
    # has an output, so the remaining pending shards should be rebalanced
    # across the (now smaller) set of tasks rather than statically hashed.
    shards = _make_shards(tmp_path, [10, 10])
    output_dir = tmp_path / "scores"
    output_dir.mkdir()
    (output_dir / f"{shards[0].stem}.parquet").touch()

    step = ScoreStep(ScoreConfig(shard_index=0, shard_count=1))
    assigned = step._get_assigned_shards(shards, output_dir)

    assert assigned == [shards[1]]


def test_score_config_defaults():
    """Test ScoreConfig has sensible defaults."""
    config = ScoreConfig()
    assert config.enabled is True
    assert config.inference_model == "flair-bio/amplify-350m"
    assert config.batch_size == 16
    assert config.mixed_precision == "bf16"
    assert config.packed is True


def test_score_collator_packs_sequences():
    """Test _ScoreCollator._pack tokenizes and packs sequences correctly."""
    tokenizer = Mock()
    tokenizer.model_max_length = int(1e6)
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    }

    collator = _ScoreCollator(
        name_column="seq_id",
        sequence_column="sequence",
        tokenizer=tokenizer,
        packed=True,
        max_length=2048,
        pad_to_multiple_of=8,
        shard_name="test",
    )

    batch_result = collator._pack(["ACGT", "AG", "ACGTAC"])

    assert "input_ids" in batch_result
    assert "cu_seqlens" in batch_result
    assert "position_ids" in batch_result
    assert batch_result["cu_seqlens"].tolist() == [0, 3, 5, 9]
    assert batch_result["max_seqlen"] == 4


def test_score_collator_pads_sequences():
    """Test _ScoreCollator._pad tokenizes and pads sequences correctly."""
    tokenizer = Mock()
    tokenizer.model_max_length = int(1e6)
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }

    collator = _ScoreCollator(
        name_column="seq_id",
        sequence_column="sequence",
        tokenizer=tokenizer,
        packed=False,
        max_length=2048,
        pad_to_multiple_of=8,
        shard_name="test",
    )

    batch_result = collator._pad(["ACGT", "AG"])

    assert "input_ids" in batch_result
    assert "attention_mask" in batch_result
    assert batch_result["cu_seqlens"] is None
    assert batch_result["max_seqlen"] is None


def test_score_batch_red_score_computation():
    """Test _score_batch returns correct output structure."""
    config = ScoreConfig(
        sequence_name_col="seq_id",
        score_column="red_score",
        packed=False,
    )
    step = ScoreStep(config)

    # Create a minimal batch with proper tensor shapes
    batch = {
        "seq_id": ["seq1", "seq2"],
        "input_ids": torch.randint(0, 100, (2, 5)).to(torch.long),
        "cu_seqlens": None,
        "attention_mask": torch.ones(2, 5, dtype=torch.bool),
        "position_ids": None,
        "max_seqlen": None,
    }

    # Create embeddings: tuple of tensors, last one is [batch, seq_len, hidden_dim]
    # This matches what model.hidden_states would return
    hidden_states_list = [torch.randn(2, 5, 64) for _ in range(3)]

    mock_output = Mock()
    mock_output.hidden_states = hidden_states_list
    mock_model = Mock(return_value=mock_output)

    result = step._score_batch(batch, mock_model, torch.device("cpu"), torch.float32)

    # Verify output structure
    assert "seq_id" in result
    assert "red_score" in result
    assert result["seq_id"] == ["seq1", "seq2"]
    assert len(result["red_score"]) == 2
    # Scores are computed as 1 - mean_sim, should be numeric and ideally in [0, 1] for normalized vectors
    assert all(isinstance(s, (int, float)) for s in result["red_score"])
