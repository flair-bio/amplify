"""Tests for modules.data.src.steps.cluster local cascade.

Coverage:
- Pure-polars tests for the cascade resumption logic in ``cascaded_mmseqs_clustering``.
- These exercise the correctness-critical join logic with no external binaries.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from modules.data.src.dataset.dataset import Dataset, DatasetConfig
from modules.data.src.steps.cluster import (
    ClusterConfig,
    ClusterStep,
    cascaded_mmseqs_clustering,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path: Path, name: str = "toy") -> Dataset:
    return Dataset(DatasetConfig(name=name, base_path=str(tmp_path / "datasets")))


def _seed_completed_round_dir(
    output_dir: Path,
    threshold_label: str,
    assignments: pl.DataFrame,
) -> None:
    """Write a completed per-round assignment directory and rep FASTA artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir = output_dir / f"clusters_{threshold_label}_assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    assignments.write_parquet(assignment_dir / "part-00000.parquet")
    (assignment_dir / "_SUCCESS").touch()
    # A non-empty representative FASTA marks the round's mmseqs output as present.
    (output_dir / f"clusters_{threshold_label}_rep_seq.fasta").write_text(">r\nACDE\n")


# ---------------------------------------------------------------------------
# ClusterStep.run path resolution (output_dir / output_path / input_fasta)
# ---------------------------------------------------------------------------


def _run_local_and_capture_call(
    monkeypatch: pytest.MonkeyPatch, step: "ClusterStep", dataset: Dataset
) -> dict:
    """Run `run` with `cascaded_mmseqs_clustering` stubbed; return its call kwargs."""
    captured: dict = {}

    def _fake_cascaded_mmseqs_clustering(**kwargs):
        captured.update(kwargs)
        return pl.LazyFrame({"sequence_id": [], "cluster_id": []})

    monkeypatch.setattr(
        "modules.data.src.steps.cluster.cascaded_mmseqs_clustering",
        _fake_cascaded_mmseqs_clustering,
    )
    step.run(dataset)
    return captured


def test_output_overrides_default_to_dataset_tmp_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no overrides configured, resolved paths fall back to dataset.tmp_path."""
    step = ClusterStep(ClusterConfig())
    dataset = _make_dataset(tmp_path)
    (dataset.tmp_path / f"{dataset.name}_all.fasta.gz").parent.mkdir(
        parents=True, exist_ok=True
    )
    fasta = dataset.tmp_path / f"{dataset.name}_all.fasta.gz"
    fasta.write_bytes(b"")

    captured = _run_local_and_capture_call(monkeypatch, step, dataset)

    assert captured["fasta_file"] == str(fasta)
    # resume defaults to False, so no persistent cascade dir is used.
    assert captured["output_dir"] is None
    assert (
        dataset.tmp_path / f"{dataset.name}_cluster_assignments.parquet"
    ).exists()


def test_output_dir_override_always_wins_regardless_of_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit output_dir override is used even when resume=False."""
    custom_dir = tmp_path / "custom_cascade_workspace"
    step = ClusterStep(ClusterConfig(output_dir=str(custom_dir), resume=False))
    dataset = _make_dataset(tmp_path)
    dataset.tmp_path.mkdir(parents=True, exist_ok=True)
    (dataset.tmp_path / f"{dataset.name}_all.fasta.gz").write_bytes(b"")

    captured = _run_local_and_capture_call(monkeypatch, step, dataset)

    assert captured["output_dir"] == str(custom_dir)
    assert custom_dir.is_dir()


def test_output_path_override_wins_over_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit output_path override replaces the dataset.tmp_path default."""
    custom_path = tmp_path / "custom_output" / "assignments.parquet"
    step = ClusterStep(ClusterConfig(output_path=str(custom_path)))
    dataset = _make_dataset(tmp_path)
    dataset.tmp_path.mkdir(parents=True, exist_ok=True)
    (dataset.tmp_path / f"{dataset.name}_all.fasta.gz").write_bytes(b"")

    _run_local_and_capture_call(monkeypatch, step, dataset)

    assert custom_path.exists()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_split_memory_limit_blank_coercion() -> None:
    """Blank CLI overrides (empty strings) are coerced to None."""
    cfg = ClusterConfig(mmseqs={"command": {"split_memory_limit": ""}})
    assert cfg.mmseqs.command.split_memory_limit is None

    cfg2 = ClusterConfig(mmseqs={"command": {"split_memory_limit": "32G"}})
    assert cfg2.mmseqs.command.split_memory_limit == "32G"


# ---------------------------------------------------------------------------
# Local-mode resume (no mmseqs: every round is pre-cached)
# ---------------------------------------------------------------------------


def test_cascaded_resume_reuses_all_cached_rounds(tmp_path: Path) -> None:
    """With every round cached, resume joins them without invoking mmseqs."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s1\nACDE\n>s2\nACDF\n>s3\nKLMN\n")
    output_dir = tmp_path / "cascade"

    _seed_completed_round_dir(
        output_dir,
        "90",
        pl.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3"],
                "cluster_rep_at_90": ["s1", "s1", "s3"],
            }
        ),
    )
    _seed_completed_round_dir(
        output_dir,
        "50",
        pl.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3"],
                "cluster_rep_at_50": ["s1", "s1", "s1"],
            }
        ),
    )

    # No mmseqs on PATH here would still pass because every round is cached.
    wide = (
        cascaded_mmseqs_clustering(
            fasta_file=str(fasta),
            identity_thresholds=[0.9, 0.5],
            output_dir=str(output_dir),
            resume=True,
        )
        .sort("sequence_id")
        .collect()
    )

    assert wide.columns == ["sequence_id", "cluster_rep_at_90", "cluster_rep_at_50"]
    assert wide["cluster_rep_at_90"].to_list() == ["s1", "s1", "s3"]
    assert wide["cluster_rep_at_50"].to_list() == ["s1", "s1", "s1"]


def test_cascaded_resume_reuses_mmseqs_output_when_assignment_missing(
    tmp_path: Path,
) -> None:
    """Partial resume: a round's MMseqs output exists but its assignment
    parquet doesn't (e.g. the Python remap/write step crashed after MMseqs
    finished). The cascade should reuse the completed cluster_tsv/rep_seq
    fasta and redo only the Python remap -- never invoking mmseqs (there is
    none on PATH in this test) -- and end up with a normal round assignment
    parquet on disk afterwards.
    """
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s1\nACDE\n>s2\nACDF\n>s3\nKLMN\n")
    output_dir = tmp_path / "cascade"

    # Round 1 (90%) fully cached, as in the fully-resumed case above.
    _seed_completed_round_dir(
        output_dir,
        "90",
        pl.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3"],
                "cluster_rep_at_90": ["s1", "s1", "s3"],
            }
        ),
    )

    # Round 2 (50%): only MMseqs outputs exist (cluster_tsv + rep_seq
    # fasta) -- the assignment parquet is deliberately absent, simulating a
    # crash between MMseqs finishing and the Python join/write completing.
    output_dir.mkdir(parents=True, exist_ok=True)
    # mmseqs cluster.tsv format: "<cluster_id>\t<sequence_id>", no header.
    # s1 and s3 (round 1's representatives) merge into a single new cluster "s1".
    (output_dir / "clusters_50_cluster.tsv").write_text("s1\ts1\ns1\ts3\n")
    (output_dir / "clusters_50_rep_seq.fasta").write_text(">s1\nACDE\n")
    assert not (output_dir / "clusters_50_assignments").exists()

    wide = cascaded_mmseqs_clustering(
        fasta_file=str(fasta),
        identity_thresholds=[0.9, 0.5],
        output_dir=str(output_dir),
        resume=True,
    ).collect().sort("sequence_id")

    assert wide.columns == ["sequence_id", "cluster_rep_at_90", "cluster_rep_at_50"]
    assert wide["cluster_rep_at_90"].to_list() == ["s1", "s1", "s3"]
    assert wide["cluster_rep_at_50"].to_list() == ["s1", "s1", "s1"]
    # The missing assignment directory should now have been written and marked complete.
    round_dir = output_dir / "clusters_50_assignments"
    assert round_dir.is_dir()
    assert (round_dir / "_SUCCESS").is_file()


def test_final_wide_join_resumes_partial_bucket_progress(tmp_path: Path) -> None:
    """If a prior run wrote some (but not all) final wide-join bucket parts
    and crashed before the `_SUCCESS` marker, a resumed run should reuse the
    already-written buckets instead of wiping `final_wide_assignments/` and
    starting over -- this is the fix for BFD ending up with an empty
    `final_wide_assignments/` directory after a preempted job.
    """
    output_dir = tmp_path / "cascade"
    _seed_completed_round_dir(
        output_dir,
        "90",
        pl.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3"],
                "cluster_rep_at_90": ["s1", "s1", "s3"],
            }
        ),
    )
    _seed_completed_round_dir(
        output_dir,
        "50",
        pl.DataFrame(
            {
                "sequence_id": ["s1", "s2", "s3"],
                "cluster_rep_at_50": ["s1", "s1", "s1"],
            }
        ),
    )

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s1\nACDE\n>s2\nACDF\n>s3\nKLMN\n")

    wide = cascaded_mmseqs_clustering(
        fasta_file=str(fasta),
        identity_thresholds=[0.9, 0.5],
        output_dir=str(output_dir),
        resume=True,
    ).collect().sort("sequence_id")

    assert wide.columns == ["sequence_id", "cluster_rep_at_90", "cluster_rep_at_50"]
    assert wide["cluster_rep_at_90"].to_list() == ["s1", "s1", "s3"]
    assert wide["cluster_rep_at_50"].to_list() == ["s1", "s1", "s1"]

    final_dir = output_dir / "final_wide_assignments"
    assert (final_dir / "_SUCCESS").is_file()

    # Second call with resume=True should short-circuit entirely: reuse the
    # completed final_wide_assignments/ dir rather than recomputing it.
    completed_at = (final_dir / "_SUCCESS").stat().st_mtime
    wide_again = cascaded_mmseqs_clustering(
        fasta_file=str(fasta),
        identity_thresholds=[0.9, 0.5],
        output_dir=str(output_dir),
        resume=True,
    ).collect().sort("sequence_id")
    assert wide_again["cluster_rep_at_90"].to_list() == ["s1", "s1", "s3"]
    assert (final_dir / "_SUCCESS").stat().st_mtime == completed_at


def test_cascaded_resume_requires_output_dir(tmp_path: Path) -> None:
    """resume=True without a persistent output_dir is a hard configuration error."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">s1\nACDE\n")
    with pytest.raises(ValueError, match="output_dir"):
        cascaded_mmseqs_clustering(
            fasta_file=str(fasta),
            identity_thresholds=[0.9],
            resume=True,
        )
