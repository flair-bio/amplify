from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import weakref

import polars as pl
import logging

from typing import Literal
from collections.abc import Generator
from pydantic import BaseModel, ConfigDict, Field, field_validator

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import (
    override_or_default,
    resolve_path,
    run_process,
)
from modules.data.src.utils.cluster_utils import (
    normalize_threshold,
    is_complete_dir,
    promote_tmp_dir,
    stage_tmp_dir,
    format_threshold_label,
    validate_num_threads,
)

logger = logging.getLogger(__name__)


class MmseqsColumnConfig(BaseModel):
    """Column names used in DataFrame transformations."""

    model_config = ConfigDict(extra="forbid")

    sequence_id: str = "sequence_id"
    cluster_id: str = "cluster_id"
    threshold_cluster_template: str = "cluster_rep_at_{threshold_label}"
    threshold_label_decimal_separator: str = "p"
    current_cluster: str = "current_cluster"
    next_cluster: str = "next_cluster"
    has_unmapped: str = "has_unmapped"


class MmseqsOutputConfig(BaseModel):
    """Filename and directory templates for MMseqs intermediate files."""

    model_config = ConfigDict(extra="forbid")

    single_result_prefix: str = "clusters"
    cascaded_result_prefix_template: str = "clusters_{threshold_label}"
    cascaded_tmp_dir_template: str = (
        "tmp_{threshold_label}"  # Hash-bucketed parquet dir with a _SUCCESS marker.
    )
    round_assignment_template: str = "clusters_{threshold_label}_assignments"
    round_assignment_by_sequence_template: str = "clusters_{threshold_label}_assignments_by_sequence"  # Same assignments, rebucketed by sequence_id for the final join.
    cluster_tsv_suffix: str = "_cluster.tsv"
    representative_fasta_suffix: str = "_rep_seq.fasta"


class MmseqsCommandConfig(BaseModel):
    """MMseqs command parameters."""

    model_config = ConfigDict(extra="forbid")

    executable: str = "mmseqs"
    workflow: Literal["easy-linclust", "easy-cluster"] = "easy-linclust"
    coverage_mode: int = Field(default=1, ge=0)
    parquet_compression: Literal[
        "lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"
    ] = "zstd"
    split_memory_limit: str | None = Field(default=None)
    clust_hash: bool = (
        False  # Enable MMseqs hash-based pre-dedup before k-mer matching.
    )
    linclust_version: Literal[1, 2] = 2  # Select MMseqs linclust algorithm version.
    # Optional flags; unset (None) values are omitted so MMseqs defaults apply.
    alignment_mode: int | None = Field(default=None, ge=0, le=4)
    seq_id_mode: int | None = Field(default=None, ge=0, le=2)
    sensitivity: float | None = Field(default=None, gt=0)
    cluster_reassign: bool | None = Field(default=None)
    max_seqs: int | None = Field(default=None, ge=1)
    kmer_per_seq: int | None = Field(default=None, ge=1)
    num_join_buckets: int = Field(
        default=32, ge=1
    )  # Bucket count for memory-bounded joins.

    @field_validator("split_memory_limit", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat blank CLI overrides (e.g. ``key=``) as unset."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class MmseqsClusteringConfig(BaseModel):
    """Top-level runtime config. for clustering helpers."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mmseqs"] = "mmseqs"

    columns: MmseqsColumnConfig = Field(default_factory=MmseqsColumnConfig)
    outputs: MmseqsOutputConfig = Field(default_factory=MmseqsOutputConfig)
    command: MmseqsCommandConfig = Field(default_factory=MmseqsCommandConfig)


class ClusterConfig(BaseModel):
    """Config for the cluster step; runs the cascade via cascaded_mmseqs_clustering()."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    identity_thresholds: list[float] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    num_threads: int = Field(default=1, ge=1)
    coverage_threshold: float = 0.8

    resume: bool = False
    input_fasta: str | Path | None = None
    output_dir: str | Path | None = None
    output_path: str | Path | None = None

    mmseqs: MmseqsClusteringConfig = Field(default_factory=MmseqsClusteringConfig)


# parsing/normalization helpers
def _scan_mmseqs_cluster_tsv(
    cluster_tsv: Path,
    cluster_column: str,
    sequence_column: str,
) -> pl.LazyFrame:
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Missing MMseqs cluster output: {cluster_tsv}")

    if cluster_tsv.stat().st_size == 0:
        return pl.LazyFrame(schema={cluster_column: pl.Utf8, sequence_column: pl.Utf8})

    # Lazy scan avoids loading full TSVs eagerly across cascade rounds.
    return pl.scan_csv(
        cluster_tsv,
        separator="\t",
        has_header=False,
        new_columns=[cluster_column, sequence_column],
        quote_char=None,
        schema_overrides={cluster_column: pl.Utf8, sequence_column: pl.Utf8},
    )


def _dedupe_join_key(frame: pl.LazyFrame, key: str, tie_break: str) -> pl.LazyFrame:
    """Keep one deterministic row per join key."""
    return frame.sort(tie_break).unique(subset=[key], keep="first")


# assignment writers/joiners
def _hash_bucket_left_join(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    on: str,
    num_buckets: int,
    dedupe_right_tie_break: str | None = None,
) -> Generator[tuple[int, pl.DataFrame, int], None, None]:
    """Yield bucketed left-join results and unmapped counts per bucket."""
    bucket_column = "_hash_bucket"
    left_bucketed = left.with_columns(
        (pl.col(on).hash() % num_buckets).alias(bucket_column)
    )
    right_bucketed = right.with_columns(
        (pl.col(on).hash() % num_buckets).alias(bucket_column)
    )
    for bucket_idx in range(num_buckets):
        left_bucket = (
            left_bucketed.filter(pl.col(bucket_column) == bucket_idx)
            .drop(bucket_column)
            .collect()
        )
        right_bucket_lazy = right_bucketed.filter(
            pl.col(bucket_column) == bucket_idx
        ).drop(bucket_column)
        if dedupe_right_tie_break is not None:
            right_bucket_lazy = _dedupe_join_key(
                right_bucket_lazy, key=on, tie_break=dedupe_right_tie_break
            )
        right_bucket = right_bucket_lazy.collect()
        unmapped_count = left_bucket.join(right_bucket, on=on, how="anti").height
        joined_bucket = left_bucket.join(right_bucket, on=on, how="left")
        yield bucket_idx, joined_bucket, unmapped_count


def _materialize_sequence_bucketed_round(
    source_assignment_dir: Path,
    output_assignment_dir: Path,
    sequence_column: str,
    num_buckets: int,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"],
    resume: bool,
) -> Path:
    """Rebucket one round's assignments by hash(sequence_id)."""
    if resume and is_complete_dir(output_assignment_dir):
        logger.info(
            "Resume: sequence-bucketed assignments already complete (%s).",
            output_assignment_dir.name,
        )
        return output_assignment_dir

    output_tmp_dir = stage_tmp_dir(output_assignment_dir, clean=not resume)

    logger.info(
        "Building sequence-bucketed assignments (%s) from %s.",
        output_assignment_dir.name,
        source_assignment_dir.name,
    )

    bucket_column = "_hash_bucket"
    for bucket_idx in range(num_buckets):
        bucket_part_path = output_tmp_dir / f"part-{bucket_idx:05d}.parquet"
        if resume and bucket_part_path.is_file():
            logger.info(
                "Resume: sequence-bucketed assignments %s bucket %s/%s already written; skipping.",
                output_assignment_dir.name,
                bucket_idx + 1,
                num_buckets,
            )
            continue
        bucket_part_tmp_path = bucket_part_path.with_name(
            bucket_part_path.name + ".writing"
        )
        (
            pl.scan_parquet(str(source_assignment_dir))
            .with_columns(
                (pl.col(sequence_column).hash() % num_buckets).alias(bucket_column)
            )
            .filter(pl.col(bucket_column) == bucket_idx)
            .drop(bucket_column)
            .sink_parquet(bucket_part_tmp_path, compression=compression)
        )
        bucket_part_tmp_path.replace(bucket_part_path)
        logger.info(
            "Sequence-bucketed assignments %s: bucket %s/%s done.",
            output_assignment_dir.name,
            bucket_idx + 1,
            num_buckets,
        )
    promote_tmp_dir(output_tmp_dir, output_assignment_dir)
    logger.info(
        "Sequence-bucketed assignments complete (%s).", output_assignment_dir.name
    )
    return output_assignment_dir


# mmseqs invocation
def _run_mmseqs_cluster(
    fasta_file: Path,
    result_prefix: Path,
    tmp_dir: Path,
    identity_threshold: float,
    coverage_threshold: float,
    num_threads: int,
    mmseqs_config: MmseqsClusteringConfig,
    split_memory_limit: str | None = None,
) -> tuple[Path, Path]:
    """Run one MMseqs clustering pass and return (cluster_tsv, rep_seq_fasta)."""
    cmd_config = mmseqs_config.command
    command = [
        cmd_config.executable,
        cmd_config.workflow,
        str(fasta_file),
        str(result_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(identity_threshold),
        "-c",
        str(coverage_threshold),
        "--cov-mode",
        str(cmd_config.coverage_mode),
        "--threads",
        str(num_threads),
    ]

    optional_flags: list[tuple[str, object | None]] = [
        ("--split-memory-limit", split_memory_limit),
        ("--alignment-mode", cmd_config.alignment_mode),
        ("--seq-id-mode", cmd_config.seq_id_mode),
        ("-s", cmd_config.sensitivity),
        (
            "--cluster-reassign",
            None
            if cmd_config.cluster_reassign is None
            else int(cmd_config.cluster_reassign),
        ),
        ("--max-seqs", cmd_config.max_seqs),
        ("--kmer-per-seq", cmd_config.kmer_per_seq),
    ]
    for flag, value in optional_flags:
        if value is not None:
            command.extend([flag, str(value)])

    # Hash-dedup near-exact sequences before k-mer matching; merged back into results.
    if cmd_config.clust_hash:
        command.extend(["--clust-hash", "1"])

    if cmd_config.workflow == "easy-linclust":
        command.extend(["--linclust-version", str(cmd_config.linclust_version)])

    run_process(command)

    # MMseqs also writes an unused "_all_seqs.fasta"; drop it to save space.
    all_seqs_fasta = result_prefix.with_name(f"{result_prefix.name}_all_seqs.fasta")
    if all_seqs_fasta.exists():
        all_seqs_fasta.unlink()

    cluster_tsv = result_prefix.with_name(
        f"{result_prefix.name}{mmseqs_config.outputs.cluster_tsv_suffix}"
    )
    rep_seq_fasta = result_prefix.with_name(
        f"{result_prefix.name}{mmseqs_config.outputs.representative_fasta_suffix}"
    )

    if not cluster_tsv.exists():
        raise FileNotFoundError(
            f"MMseqs completed but did not produce expected cluster file: {cluster_tsv}"
        )
    if not rep_seq_fasta.exists():
        raise FileNotFoundError(
            f"MMseqs completed but did not produce expected representative fasta: {rep_seq_fasta}"
        )

    return cluster_tsv, rep_seq_fasta


# public entrypoints
def mmseqs_clustering(
    fasta_file: str,
    identity_threshold: float,
    coverage_threshold: float,
    num_threads: int = 1,
    mmseqs_config: MmseqsClusteringConfig | None = None,
) -> pl.DataFrame:
    """Run a single MMseqs2 clustering pass and return cluster assignments.

    Args:
        fasta_file: Path to the input FASTA file.
        identity_threshold: Min sequence identity; fraction (0.9) or percent (90) form.
        coverage_threshold: Min coverage; fraction (0.8) or percent (80) form.
        num_threads: Number of threads to use.
    Returns:
        A DataFrame with 'sequence_id' and 'cluster_id' columns.
    """
    resolved_mmseqs_config = mmseqs_config or MmseqsClusteringConfig()
    cols = resolved_mmseqs_config.columns
    outs = resolved_mmseqs_config.outputs

    fasta_path = Path(fasta_file)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Input fasta file does not exist: {fasta_path}")

    normalized_identity_threshold = normalize_threshold(
        "identity_threshold",
        identity_threshold,
    )
    normalized_coverage_threshold = normalize_threshold(
        "coverage_threshold",
        coverage_threshold,
    )
    validate_num_threads(num_threads)

    # Temporary workspace keeps intermediate MMseqs artifacts out of the repo.
    with TemporaryDirectory(prefix="mmseqs_cluster_") as tmp_root:
        tmp_root_path = Path(tmp_root)
        result_prefix = tmp_root_path / outs.single_result_prefix

        cluster_tsv, _ = _run_mmseqs_cluster(
            fasta_file=fasta_path,
            result_prefix=result_prefix,
            tmp_dir=tmp_root_path / "tmp",
            identity_threshold=normalized_identity_threshold,
            coverage_threshold=normalized_coverage_threshold,
            num_threads=num_threads,
            mmseqs_config=resolved_mmseqs_config,
        )

        assignments = _scan_mmseqs_cluster_tsv(
            cluster_tsv,
            cluster_column=cols.cluster_id,
            sequence_column=cols.sequence_id,
        ).select(
            cols.sequence_id,
            cols.cluster_id,
        )
        return assignments.collect()


@dataclass(frozen=True)
class RoundPaths:
    """Every artifact path/column name used by one cascade round."""

    threshold_label: str
    threshold_cluster_column: str
    result_prefix: Path
    round_assignment_path: Path
    round_assignment_by_sequence_path: Path
    rep_seq_fasta: Path
    cluster_tsv: Path
    tmp_dir: Path


def _resolve_round_paths(
    round_idx: int,
    raw_threshold: float,
    tmp_root_path: Path,
    mmseqs_config: MmseqsClusteringConfig,
) -> RoundPaths:
    """Compute every artifact path/column name used by one cascade round."""
    columns = mmseqs_config.columns
    outputs = mmseqs_config.outputs
    threshold_label = format_threshold_label(
        raw_threshold, columns.threshold_label_decimal_separator
    )
    fmt_kwargs = {
        "round_idx": round_idx,
        "threshold": raw_threshold,
        "threshold_label": threshold_label,
    }
    result_prefix = tmp_root_path / outputs.cascaded_result_prefix_template.format(
        **fmt_kwargs
    )
    return RoundPaths(
        threshold_label=threshold_label,
        threshold_cluster_column=columns.threshold_cluster_template.format(
            **fmt_kwargs
        ),
        result_prefix=result_prefix,
        round_assignment_path=tmp_root_path
        / outputs.round_assignment_template.format(**fmt_kwargs),
        round_assignment_by_sequence_path=tmp_root_path
        / outputs.round_assignment_by_sequence_template.format(**fmt_kwargs),
        rep_seq_fasta=result_prefix.with_name(
            f"{result_prefix.name}{outputs.representative_fasta_suffix}"
        ),
        cluster_tsv=result_prefix.with_name(
            f"{result_prefix.name}{outputs.cluster_tsv_suffix}"
        ),
        tmp_dir=tmp_root_path / outputs.cascaded_tmp_dir_template.format(**fmt_kwargs),
    )


def _try_resume_round(
    round_idx: int,
    raw_threshold: float,
    paths: RoundPaths,
    columns: MmseqsColumnConfig,
) -> pl.LazyFrame | None:
    """Return cached membership if this round's assignment and rep FASTA are already complete."""
    round_assignment_path = paths.round_assignment_path
    rep_seq_fasta = paths.rep_seq_fasta
    threshold_cluster_column = paths.threshold_cluster_column

    if not (is_complete_dir(round_assignment_path) and rep_seq_fasta.is_file()):
        return None
    logger.info(
        "Resume: round %s (threshold %s) already complete; reusing cached assignment.",
        round_idx,
        raw_threshold,
    )
    return pl.scan_parquet(round_assignment_path).rename(
        {threshold_cluster_column: columns.current_cluster}
    )


# assignment writers/joiners
def _write_round_one_assignment(
    current_round: pl.LazyFrame,
    columns: MmseqsColumnConfig,
    threshold_cluster_column: str,
    round_assignment_tmp_dir: Path,
    compression: str,
) -> pl.LazyFrame:
    """Round 1 maps original sequences directly onto their clusters."""
    membership_by_original = current_round.rename(
        {columns.cluster_id: columns.current_cluster}
    )
    threshold_assignments = current_round.select(
        columns.sequence_id,
        pl.col(columns.cluster_id).alias(threshold_cluster_column),
    )
    threshold_assignments.sink_parquet(
        round_assignment_tmp_dir / "part-00000.parquet",
        compression=compression,
    )
    return membership_by_original


def _write_remap_round_assignment(
    membership_by_original: pl.LazyFrame,
    current_round: pl.LazyFrame,
    columns: MmseqsColumnConfig,
    threshold_cluster_column: str,
    round_assignment_tmp_dir: Path,
    num_join_buckets: int,
    compression: str,
    round_idx: int,
    raw_threshold: float,
) -> None:
    """Remap the prior round's clusters onto this round's clusters via bucketed joins."""
    remap = current_round.rename(
        {
            columns.sequence_id: columns.current_cluster,
            columns.cluster_id: columns.next_cluster,
        }
    )
    # Bucketed join bounds memory and dedupes remap keys per bucket.
    total_unmapped = 0
    for bucket_idx, joined_bucket, unmapped_count in _hash_bucket_left_join(
        membership_by_original,
        remap,
        on=columns.current_cluster,
        num_buckets=num_join_buckets,
        dedupe_right_tie_break=columns.next_cluster,
    ):
        total_unmapped += unmapped_count
        bucket_assignments = joined_bucket.select(
            pl.col(columns.sequence_id),
            pl.col(columns.next_cluster).alias(threshold_cluster_column),
        )
        bucket_assignments.write_parquet(
            round_assignment_tmp_dir / f"part-{bucket_idx:05d}.parquet",
            compression=compression,
        )
        logger.info(
            "Round %s (threshold %s): bucket %s/%s done (%s rows, %s unmapped).",
            round_idx,
            raw_threshold,
            bucket_idx + 1,
            num_join_buckets,
            bucket_assignments.height,
            unmapped_count,
        )

    if total_unmapped:
        raise RuntimeError(
            "Failed to map some sequences to cascaded clusters in round "
            f"{round_idx} ({total_unmapped} unmapped rows across "
            f"{num_join_buckets} buckets)."
        )


def _write_final_wide_bucket(
    bucket_idx: int,
    num_final_buckets: int,
    sequence_bucketed_paths: list[Path],
    columns: MmseqsColumnConfig,
    bucket_part_path: Path,
    compression: str,
) -> None:
    """Join one sequence_id bucket across all rounds and write it to disk.

    Each round's result is deduped back to one row per sequence_id to prevent duplicate keys compounding across
    rounds.
    """
    wide_bucket: pl.DataFrame | None = None
    num_rounds = len(sequence_bucketed_paths)
    for round_position, assignment_by_sequence_path in enumerate(
        sequence_bucketed_paths, start=1
    ):
        logger.info(
            "Final wide join: bucket %s/%s -- joining round %s/%s (%s).",
            bucket_idx + 1,
            num_final_buckets,
            round_position,
            num_rounds,
            assignment_by_sequence_path.name,
        )
        round_bucket_part_path = (
            assignment_by_sequence_path / f"part-{bucket_idx:05d}.parquet"
        )
        round_bucket = pl.scan_parquet(round_bucket_part_path).collect()
        wide_bucket = (
            round_bucket
            if wide_bucket is None
            else wide_bucket.join(round_bucket, on=columns.sequence_id, how="left")
        )
        round_cluster_column = next(
            column for column in round_bucket.columns if column != columns.sequence_id
        )
        wide_bucket = wide_bucket.sort(
            [columns.sequence_id, round_cluster_column]
        ).unique(subset=[columns.sequence_id], keep="first")

    assert wide_bucket is not None
    logger.info(
        "Final wide join: bucket %s/%s -- join complete (%s rows), writing to disk...",
        bucket_idx + 1,
        num_final_buckets,
        wide_bucket.height,
    )
    bucket_part_tmp_path = bucket_part_path.with_name(
        bucket_part_path.name + ".writing"
    )
    wide_bucket.write_parquet(bucket_part_tmp_path, compression=compression)
    bucket_part_tmp_path.replace(bucket_part_path)
    bucket_row_count = (
        pl.scan_parquet(bucket_part_path).select(pl.len()).collect().item()
    )
    logger.info(
        "Final wide join: bucket %s/%s done (%s rows).",
        bucket_idx + 1,
        num_final_buckets,
        bucket_row_count,
    )


def _build_final_wide_assignments(
    tmp_root_path: Path,
    threshold_assignment_paths: list[Path],
    threshold_assignment_by_sequence_paths: list[Path],
    resolved_mmseqs_config: MmseqsClusteringConfig,
    resume: bool,
) -> pl.LazyFrame:
    """Join every round's per-sequence assignment into one wide table, bucketed to bound memory."""
    cols = resolved_mmseqs_config.columns
    cmd = resolved_mmseqs_config.command
    num_final_buckets = cmd.num_join_buckets
    compression = cmd.parquet_compression
    final_output_dir = tmp_root_path / "final_wide_assignments"

    if resume and is_complete_dir(final_output_dir):
        logger.info("Resume: final wide join already complete; reusing cached output.")
        return pl.scan_parquet(final_output_dir)

    # Rebucket each round's assignments by sequence_id to align the final join.
    sequence_bucketed_paths: list[Path] = []
    for round_idx, (
        round_assignment_path,
        round_assignment_by_sequence_path,
    ) in enumerate(
        zip(threshold_assignment_paths, threshold_assignment_by_sequence_paths),
        start=1,
    ):
        logger.info(
            "Preparing sequence-bucketed assignments for round %s (%s).",
            round_idx,
            round_assignment_by_sequence_path.name,
        )
        sequence_bucketed_paths.append(
            _materialize_sequence_bucketed_round(
                source_assignment_dir=round_assignment_path,
                output_assignment_dir=round_assignment_by_sequence_path,
                sequence_column=cols.sequence_id,
                num_buckets=num_final_buckets,
                compression=compression,
                resume=resume,
            )
        )

    final_output_tmp_dir = stage_tmp_dir(final_output_dir, clean=not resume)

    for bucket_idx in range(num_final_buckets):
        bucket_part_path = final_output_tmp_dir / f"part-{bucket_idx:05d}.parquet"
        if resume and bucket_part_path.is_file():
            logger.info(
                "Resume: final wide join bucket %s/%s already written; skipping.",
                bucket_idx + 1,
                num_final_buckets,
            )
            continue
        _write_final_wide_bucket(
            bucket_idx,
            num_final_buckets,
            sequence_bucketed_paths,
            cols,
            bucket_part_path,
            compression,
        )

    promote_tmp_dir(final_output_tmp_dir, final_output_dir)
    return pl.scan_parquet(final_output_dir)


def _prepare_cascade_setup(
    fasta_file: str,
    identity_thresholds: list[float],
    num_threads: int,
    coverage_threshold: float,
    output_dir: str | Path | None,
    mmseqs_config: MmseqsClusteringConfig | None,
    resume: bool,
) -> tuple[
    MmseqsClusteringConfig,
    MmseqsColumnConfig,
    Path,
    list[float],
    float,
    AbstractContextManager[str],
]:
    """Resolve config and validate all user-facing cascade inputs."""
    resolved_mmseqs_config = mmseqs_config or MmseqsClusteringConfig()
    cols = resolved_mmseqs_config.columns

    fasta_path = Path(fasta_file)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"Input fasta file does not exist: {fasta_path}")
    if not identity_thresholds:
        raise ValueError("identity_thresholds cannot be empty.")
    if resume and output_dir is None:
        raise ValueError(
            "resume=True requires output_dir so cascade artifacts persist across restarts."
        )

    normalized_identity_thresholds = [
        normalize_threshold("identity_threshold", t) for t in identity_thresholds
    ]
    normalized_coverage_threshold = normalize_threshold(
        "coverage_threshold", coverage_threshold
    )
    validate_num_threads(num_threads)

    # Isolate artifacts in a temp workspace by default; output_dir opts into persistence.
    workspace_context: AbstractContextManager[str] = TemporaryDirectory(
        prefix="mmseqs_cascade_"
    )
    if output_dir is not None:
        persistent_output_root = Path(output_dir)
        persistent_output_root.mkdir(parents=True, exist_ok=True)
        workspace_context = nullcontext(str(persistent_output_root))

    return (
        resolved_mmseqs_config,
        cols,
        fasta_path,
        normalized_identity_thresholds,
        normalized_coverage_threshold,
        workspace_context,
    )


def _run_cascade_rounds(
    *,
    tmp_root_path: Path,
    fasta_path: Path,
    identity_thresholds: list[float],
    normalized_identity_thresholds: list[float],
    normalized_coverage_threshold: float,
    num_threads: int,
    split_memory_limit: str | None,
    resolved_mmseqs_config: MmseqsClusteringConfig,
    cols: MmseqsColumnConfig,
    resume: bool,
) -> tuple[list[Path], list[Path]]:
    """Run/restore all cascade rounds and materialize round assignment artifacts."""
    cmd = resolved_mmseqs_config.command
    # current_fasta starts as the input FASTA, then becomes each round's representative sequences.
    current_fasta = fasta_path
    # Maps each original sequence to its cluster ID at the current cascade depth.
    membership_by_original: pl.LazyFrame | None = None

    threshold_assignment_paths: list[Path] = []
    threshold_assignment_by_sequence_paths: list[Path] = []

    for round_idx, (raw_identity_threshold, normalized_identity_threshold) in enumerate(
        zip(identity_thresholds, normalized_identity_thresholds),
        start=1,
    ):
        paths = _resolve_round_paths(
            round_idx,
            raw_identity_threshold,
            tmp_root_path,
            resolved_mmseqs_config,
        )

        cached_membership = (
            _try_resume_round(round_idx, raw_identity_threshold, paths, cols)
            if resume
            else None
        )

        round_assignment_path = paths.round_assignment_path
        round_assignment_by_sequence_path = paths.round_assignment_by_sequence_path
        rep_seq_fasta = paths.rep_seq_fasta
        cluster_tsv_path = paths.cluster_tsv
        result_prefix = paths.result_prefix
        tmp_dir = paths.tmp_dir
        threshold_cluster_column = paths.threshold_cluster_column

        if cached_membership is not None:
            membership_by_original = cached_membership
            threshold_assignment_paths.append(round_assignment_path)
            threshold_assignment_by_sequence_paths.append(
                round_assignment_by_sequence_path
            )
            current_fasta = rep_seq_fasta
            continue

        # Partial resume: reuse MMseqs outputs and rerun the remap/write step only.
        mmseqs_already_done = (
            resume and cluster_tsv_path.is_file() and rep_seq_fasta.is_file()
        )
        cluster_tsv, next_fasta = cluster_tsv_path, rep_seq_fasta

        try:
            if mmseqs_already_done:
                logger.info(
                    "Resume: round %s (threshold %s) MMseqs already complete; "
                    "skipping mmseqs and reusing cluster_tsv for the remap step only.",
                    round_idx,
                    raw_identity_threshold,
                )
            else:
                cluster_tsv, next_fasta = _run_mmseqs_cluster(
                    fasta_file=current_fasta,
                    result_prefix=result_prefix,
                    tmp_dir=tmp_dir,
                    identity_threshold=normalized_identity_threshold,
                    coverage_threshold=normalized_coverage_threshold,
                    num_threads=num_threads,
                    mmseqs_config=resolved_mmseqs_config,
                    split_memory_limit=split_memory_limit,
                )

            current_round = _scan_mmseqs_cluster_tsv(
                cluster_tsv,
                cluster_column=cols.cluster_id,
                sequence_column=cols.sequence_id,
            ).select(
                cols.sequence_id,
                cols.cluster_id,
            )

            round_assignment_tmp_dir = stage_tmp_dir(round_assignment_path)

            if round_idx == 1:
                membership_by_original = _write_round_one_assignment(
                    current_round,
                    cols,
                    threshold_cluster_column,
                    round_assignment_tmp_dir,
                    cmd.parquet_compression,
                )
            else:
                if membership_by_original is None:
                    raise RuntimeError(
                        "Unexpected empty cluster state in cascaded run."
                    )
                _write_remap_round_assignment(
                    membership_by_original,
                    current_round,
                    cols,
                    threshold_cluster_column,
                    round_assignment_tmp_dir,
                    cmd.num_join_buckets,
                    cmd.parquet_compression,
                    round_idx,
                    raw_identity_threshold,
                )

            # Write atomically so resume only sees fully completed rounds.
            promote_tmp_dir(round_assignment_tmp_dir, round_assignment_path)
            threshold_assignment_paths.append(round_assignment_path)
            threshold_assignment_by_sequence_paths.append(
                round_assignment_by_sequence_path
            )

            # Re-scan round assignments lazily from disk for the next round.
            membership_by_original = pl.scan_parquet(round_assignment_path).rename(
                {threshold_cluster_column: cols.current_cluster}
            )
            current_fasta = next_fasta
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    return threshold_assignment_paths, threshold_assignment_by_sequence_paths


# public entrypoints
def cascaded_mmseqs_clustering(
    fasta_file: str,
    identity_thresholds: list[float],
    num_threads: int = 1,
    coverage_threshold: float = 0.8,
    split_memory_limit: str | None = None,
    output_dir: str | Path | None = None,
    mmseqs_config: MmseqsClusteringConfig | None = None,
    resume: bool = False,
) -> pl.LazyFrame:
    """Run cascaded MMseqs2 clustering across multiple identity thresholds.

    Each round clusters the previous round's representative sequences at a
    looser threshold, remapping every original sequence to its cluster at
    each depth. Requires MMseqs2 on PATH and a well-formed input FASTA.

    Args:
        fasta_file: Path to the input FASTA file.
        identity_thresholds: Min identity per round; fraction ([0.9, 0.8]) or
            percent ([90, 80, 30]) form.
        num_threads: Number of threads to use.
        coverage_threshold: Min aligned coverage, applied at every round;
            fraction (0.8) or percent (80) form.
        split_memory_limit: Optional mmseqs ``--split-memory-limit`` value (to avoid OOM).
        output_dir: Persist cascade artifacts here instead of a temporary
            workspace. Required when ``resume`` is ``True``.
        mmseqs_config: Optional pre-built clustering config.
        resume: When ``True``, reuse completed per-round artifacts in
            ``output_dir`` and only re-run incomplete rounds. A round counts
            as complete once its assignment Parquet and representative FASTA
            both exist (e.g. after a SLURM preemption).
    Returns:
        A LazyFrame with one row per input sequence and one cluster column
        per threshold (named via ``columns.threshold_cluster_template``).
        Returned lazily so callers can stream it to disk via ``sink_parquet``
        without materializing the full result in memory.
    """
    (
        resolved_mmseqs_config,
        cols,
        fasta_path,
        normalized_identity_thresholds,
        normalized_coverage_threshold,
        workspace_context,
    ) = _prepare_cascade_setup(
        fasta_file=fasta_file,
        identity_thresholds=identity_thresholds,
        num_threads=num_threads,
        coverage_threshold=coverage_threshold,
        output_dir=output_dir,
        mmseqs_config=mmseqs_config,
        resume=resume,
    )

    # Enter the workspace context manually (rather than via `with`) so that,
    # for an ephemeral TemporaryDirectory, its cleanup can be deferred until
    # the returned LazyFrame is no longer needed. Exiting the context before
    # returning would delete the on-disk Parquet files this lazy frame scans,
    # breaking callers that stream the result via `sink_parquet` afterward.
    workspace_root = workspace_context.__enter__()
    try:
        tmp_root_path = Path(workspace_root)
        threshold_assignment_paths, threshold_assignment_by_sequence_paths = (
            _run_cascade_rounds(
                tmp_root_path=tmp_root_path,
                fasta_path=fasta_path,
                identity_thresholds=identity_thresholds,
                normalized_identity_thresholds=normalized_identity_thresholds,
                normalized_coverage_threshold=normalized_coverage_threshold,
                num_threads=num_threads,
                split_memory_limit=split_memory_limit,
                resolved_mmseqs_config=resolved_mmseqs_config,
                cols=cols,
                resume=resume,
            )
        )

        result = _build_final_wide_assignments(
            tmp_root_path,
            threshold_assignment_paths,
            threshold_assignment_by_sequence_paths,
            resolved_mmseqs_config,
            resume,
        )
    except BaseException:
        workspace_context.__exit__(*sys.exc_info())
        raise

    # Tie workspace cleanup to the lifetime of the returned LazyFrame instead
    # of the function call, so the underlying Parquet files stay on disk
    # until the caller has finished reading from it (e.g. via sink_parquet).
    weakref.finalize(result, workspace_context.__exit__, None, None, None)
    return result


# public entrypoints
class ClusterStep:
    """Pipeline step for clustering sequences.

    Performs cascaded clustering using MMseqs2 at the configured identity
    thresholds in a single process. Final output is a wide
    ``{name}_cluster_assignments.parquet`` with one cluster column per threshold.
    """

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config

    def run(self, dataset: Dataset) -> None:
        dataset.setup_directories()

        # 1. Get input FASTA path
        fasta_path = resolve_path(
            override=self.config.input_fasta,
            default=dataset.tmp_path / f"{dataset.name}_all.fasta.gz",
            label="Input FASTA",
            required=True,
        )
        assert fasta_path is not None
        assert fasta_path.is_file(), f"Input FASTA file does not exist: {fasta_path}"

        # 2. Resolve workspace directory for intermediate cascade artifacts.
        workspace_dir = None
        if self.config.output_dir or self.config.resume:
            workspace_dir = override_or_default(
                override=self.config.output_dir,
                default=self._local_cascade_dir(dataset),
            )
            workspace_dir.mkdir(parents=True, exist_ok=True)

        # 3. Resolve final parquet output path.
        output_path = override_or_default(
            override=self.config.output_path,
            default=dataset.tmp_path / f"{dataset.name}_cluster_assignments.parquet",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Run cascaded MMseqs clustering
        logger.info(
            "Running cascaded MMseqs clustering on %s with thresholds %s.",
            fasta_path,
            self.config.identity_thresholds,
        )
        cluster_assignments = cascaded_mmseqs_clustering(
            fasta_file=str(fasta_path),
            identity_thresholds=self.config.identity_thresholds,
            num_threads=self.config.num_threads,
            coverage_threshold=self.config.coverage_threshold,
            split_memory_limit=self.config.mmseqs.command.split_memory_limit,
            mmseqs_config=self.config.mmseqs,
            output_dir=str(workspace_dir) if workspace_dir is not None else None,
            resume=self.config.resume,
        )
        # 5. Stream final output directly to parquet.
        logger.info("Writing final cluster assignments to %s.", output_path)
        cluster_assignments.sink_parquet(output_path)

    def _local_cascade_dir(self, dataset: Dataset) -> Path:
        """Default persistent workspace for resumable local-mode cascade artifacts."""
        return dataset.tmp_path / f"{dataset.name}_local_cascade"
