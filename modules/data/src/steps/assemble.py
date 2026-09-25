import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import (
    override_or_default,
    resolve_path,
)

logger = logging.getLogger(__name__)


class AssembleConfig(BaseModel):
    """Config for the assemble step."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    base_id_column: str = "sequence_id"  # ID column names that join all parquet files
    cluster_id_column: str = "sequence_id"  # ID column in the cluster parquet files
    score_id_column: str = "sequence_id"  # ID column in the score parquet files
    dataset_id: str = ""  # Data source identifier (e.g. "UniRef100", "BFD", "MGnify").

    skip_existing: bool = True  # Lets a rerun pick up where a previous run where completed files are skipped
    overwrite: bool = False  # Overwrite forces full reprocessing regardless.
    max_workers: int | None = None  # Thread pool size for sharded assembly
    num_join_buckets: int = 32

    # Optional path overrides
    base_source: str | None = (
        None  # Path to a directory of shards or a single parquet file with sequence id and sequence columns
    )
    output_dir: str | None = None  # Directory where assembled output will be written
    output_file: str | None = None  # File name for the assembled output
    cluster_source: str | None = (
        None  # Path to a directory of cluster shards or a single parquet file
    )
    score_source: str | None = (
        None  # Path to a directory of score shards or a single parquet file
    )


def _bucket_cluster_source(
    source: Path, id_column: str, num_buckets: int, out_dir: Path
) -> Path:
    """Split the cluster parquet into num_buckets files hashed by id_column, reading it once."""
    if len(list(out_dir.glob("part_*.parquet"))) == num_buckets:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(source)
    for i in range(num_buckets):
        lf.filter(pl.col(id_column).hash() % num_buckets == i).sink_parquet(
            out_dir / f"part_{i:04d}.parquet"
        )
    return out_dir


def _join_cluster_bucket(
    bucket_idx: int,
    num_buckets: int,
    cluster_bucket_path: Path,
    cluster_id_column: str,
    base_id_column: str,
    shards: list[Path],
    parts_dir: Path,
) -> None:
    """Load one cluster bucket once and join it against every shard's matching rows."""
    pending_shards = []
    skipped = 0
    for shard in shards:
        shard_parts_dir = parts_dir / shard.stem
        out_part_path = shard_parts_dir / f"part_{bucket_idx:04d}.parquet"
        if out_part_path.exists():
            try:
                # Cheap validity check: read just the footer/schema, not the data.
                pl.scan_parquet(out_part_path).limit(0).collect()
                skipped += 1
                continue
            except Exception:
                logger.warning(
                    "Bucket %d/%d: found corrupted part file %s — redoing it.",
                    bucket_idx,
                    num_buckets,
                    out_part_path,
                )
                out_part_path.unlink()
        pending_shards.append(shard)

    if not pending_shards:
        logger.info(
            "Bucket %d/%d: all %d shard(s) already done — skipping.",
            bucket_idx,
            num_buckets,
            skipped,
        )
        return

    logger.info(
        "Bucket %d/%d: %d shard(s) already done (skipped), %d shard(s) pending...",
        bucket_idx,
        num_buckets,
        skipped,
        len(pending_shards),
    )

    cluster_df = (
        pl.scan_parquet(cluster_bucket_path)
        .rename({cluster_id_column: base_id_column})
        .collect()
    )
    bucket_expr = pl.col(base_id_column).hash() % num_buckets
    for shard in pending_shards:
        shard_bucket = (
            pl.scan_parquet(shard).filter(bucket_expr == bucket_idx).collect()
        )
        joined = shard_bucket.join(cluster_df, on=base_id_column, how="inner")
        shard_parts_dir = parts_dir / shard.stem
        shard_parts_dir.mkdir(parents=True, exist_ok=True)
        out_part_path = shard_parts_dir / f"part_{bucket_idx:04d}.parquet"
        tmp_part_path = out_part_path.with_suffix(".parquet.tmp")
        joined.write_parquet(tmp_part_path)
        tmp_part_path.rename(out_part_path)

    logger.info(
        "Bucket %d/%d: complete (%d shard(s) processed, %d skipped).",
        bucket_idx,
        num_buckets,
        len(pending_shards),
        skipped,
    )


def _finalize_shard(
    shard: Path,
    out_path: Path,
    parts_dir: Path,
    dataset_id: str,
    base_id_column: str,
    score_id_column: str,
    score_source: Path | None,
) -> Path:
    """Concat one shard's cluster-joined bucket parts, add metadata, and join scores."""
    shard_lf = pl.scan_parquet(parts_dir / shard.stem).with_columns(
        pl.lit(dataset_id).alias("dataset_id"),
        pl.col("sequence").str.len_chars().cast(pl.UInt32).alias("sequence_length"),
    )

    if score_source is not None:
        score_path = (
            score_source / shard.name if score_source.is_dir() else score_source
        )
        if not score_path.exists():
            logger.warning(
                "No matching score shard found for %s — skipping score join.",
                shard.name,
            )
        else:
            score_lf = pl.scan_parquet(score_path).rename(
                {score_id_column: base_id_column}
            )
            shard_lf = shard_lf.join(score_lf, on=base_id_column, how="inner")

    tmp_path = out_path.with_suffix(".parquet.tmp")
    shard_lf.sink_parquet(str(tmp_path), row_group_size=100_000)
    tmp_path.rename(out_path)
    return out_path


def assemble_data(
    base_source: Path,
    output_path: Path,
    work_dir: Path,
    dataset_id: str,
    cluster_source: Path,
    score_source: Path | None = None,
    base_id_column: str = "sequence_id",
    cluster_id_column: str = "sequence_id",
    score_id_column: str = "sequence_id",
    skip_existing: bool = True,
    overwrite: bool = False,
    max_workers: int | None = None,
    num_join_buckets: int = 32,
) -> list[Path]:
    """Assemble parquet files (either a single file or a directory of shards)."""

    is_single_file = base_source.is_file()
    shards = [base_source] if is_single_file else sorted(base_source.glob("*.parquet"))

    if not shards:
        raise FileNotFoundError(f"No parquet files found in {base_source}")

    # target_dir holds only the final output; work_dir holds the intermediate
    # cluster buckets and per-shard join parts.
    target_dir = output_path.parent if is_single_file else output_path
    target_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    for stale_tmp in (
        *target_dir.glob("*.parquet.tmp"),
        *work_dir.glob("**/*.parquet.tmp"),
    ):
        stale_tmp.unlink()

    do_skip = skip_existing and not overwrite
    pending: list[tuple[Path, Path]] = []
    output_paths: list[Path] = []

    for shard in shards:
        out_file = output_path if is_single_file else output_path / shard.name
        if do_skip and out_file.exists():
            output_paths.append(out_file)
        else:
            pending.append((shard, out_file))

    if not pending:
        logger.info(
            "All %d file(s) already assembled in %s — nothing to do.",
            len(shards),
            target_dir,
        )
        return sorted(output_paths)

    logger.info(
        "Assembling %d pending file(s) (out of %d total) -> %s",
        len(pending),
        len(shards),
        target_dir,
    )

    # Use "spawn" instead of the platform default ("fork" on Linux) to avoid deadlocks:
    # Polars/Rayon initializes internal worker threads on import, and forking a
    # multi-threaded process can deadlock the child before it executes any code.
    mp_context = mp.get_context("spawn")

    # Bucket the cluster table once, then join bucket-by-bucket against every shard:
    # the cluster table is read once total instead of once per shard, and each join
    # is bounded to one bucket's worth of data.
    logger.info("Bucketing cluster source into %d bucket(s)...", num_join_buckets)
    cluster_buckets_dir = _bucket_cluster_source(
        cluster_source,
        cluster_id_column,
        num_join_buckets,
        work_dir / "_cluster_buckets",
    )
    logger.info("Cluster source bucketed at %s.", cluster_buckets_dir)
    parts_dir = work_dir / "_assemble_parts"
    pending_shards = [shard for shard, _ in pending]

    logger.info(
        "Phase 1/2: joining %d cluster bucket(s) against %d shard(s)...",
        num_join_buckets,
        len(pending_shards),
    )
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context
    ) as executor:
        bucket_futures = {
            executor.submit(
                _join_cluster_bucket,
                i,
                num_join_buckets,
                cluster_buckets_dir / f"part_{i:04d}.parquet",
                cluster_id_column,
                base_id_column,
                pending_shards,
                parts_dir,
            ): i
            for i in range(num_join_buckets)
        }
        completed_buckets = 0
        for future in as_completed(bucket_futures):
            bucket_idx = bucket_futures[future]
            future.result()
            completed_buckets += 1
            logger.info(
                "Bucket join progress: bucket %d/%d finished (%d/%d bucket(s) done overall)",
                bucket_idx,
                num_join_buckets,
                completed_buckets,
                num_join_buckets,
            )
    logger.info("Phase 1/2 complete: all cluster buckets joined against shards.")

    logger.info(
        "Phase 2/2: finalizing %d shard(s) (concat parts, add metadata, join scores)...",
        len(pending),
    )
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context
    ) as executor:
        finalize_futures = {
            executor.submit(
                _finalize_shard,
                shard,
                out_file,
                parts_dir,
                dataset_id,
                base_id_column,
                score_id_column,
                score_source,
            ): shard
            for shard, out_file in pending
        }
        completed_shards = 0
        for finalize_future in as_completed(finalize_futures):
            shard_path = finalize_futures[finalize_future]
            try:
                written = finalize_future.result()
                completed_shards += 1
                logger.info(
                    "Successfully wrote %s (%d/%d shard(s) finalized)",
                    written.name,
                    completed_shards,
                    len(pending),
                )
                output_paths.append(written)
            except Exception:
                logger.exception("Failed to assemble shard %s", shard_path)
                raise
    logger.info("Phase 2/2 complete: all shards finalized.")

    return sorted(output_paths)


class AssembleStep:
    """Pipeline step that assembles the final training parquet files or shards."""

    def __init__(self, config: AssembleConfig) -> None:
        self.config = config

    def run(self, dataset: Dataset) -> None:
        dataset.setup_directories()

        base_path = resolve_path(
            self.config.base_source,
            dataset.tmp_path / f"{dataset.name}_parquet_shards",
            "Base",
            required=True,
        )
        assert base_path is not None

        cluster_path = resolve_path(
            self.config.cluster_source,
            dataset.tmp_path / f"{dataset.name}_cluster_assignments.parquet",
            "Cluster",
            required=True,
        )
        assert cluster_path is not None

        score_path = resolve_path(
            self.config.score_source,
            dataset.tmp_path / f"{dataset.name}_scores",
            "Score",
            required=False,
        )

        # Determine if we are outputting to a file or a directory based on input
        if base_path.is_file():
            output_path = override_or_default(
                self.config.output_file,
                dataset.train_path / f"{dataset.name}_assembled.parquet",
            )
        else:
            output_path = override_or_default(
                self.config.output_dir,
                dataset.train_path / f"{dataset.name}_assembled",
            )

        logger.info(
            "Starting assembly | base=%s | clusters=%s | scores=%s → %s",
            base_path,
            cluster_path,
            score_path or "skipped",
            output_path,
        )

        assemble_data(
            base_source=base_path,
            output_path=output_path,
            work_dir=dataset.tmp_path / f"{dataset.name}_assemble_work",
            dataset_id=self.config.dataset_id,
            cluster_source=cluster_path,
            score_source=score_path,
            base_id_column=self.config.base_id_column,
            cluster_id_column=self.config.cluster_id_column,
            score_id_column=self.config.score_id_column,
            skip_existing=self.config.skip_existing,
            overwrite=self.config.overwrite,
            max_workers=self.config.max_workers,
            num_join_buckets=self.config.num_join_buckets,
        )

        logger.info("Assembly complete.")
