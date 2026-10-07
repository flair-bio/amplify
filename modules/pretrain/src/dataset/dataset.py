"""Data sources: loading, cluster-based train/val split, indexing, and epoch plans."""

import glob
import hashlib
import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types as pa_types
from datasets import Dataset as HFDataset  # type: ignore[attr-defined]
from datasets import load_dataset, load_from_disk  # type: ignore[attr-defined]
from datasets.config import HF_DATASETS_CACHE
from huggingface_hub.constants import HF_HUB_CACHE
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.core.utils.madvise import MADV_RANDOM, advise_available, advise_paths
from modules.core.utils.units import Bytes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DataSourceConfig(BaseModel):
    """A single configured data source (local Parquet, HF Hub, or an exported cache)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["parquet", "hub", "prepared"] = Field(
        description="Which loader to use. `prepared` loads a folder written by "
        "`dataset.export_dir`, skipping the load and split entirely."
    )
    sampling_fraction: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=("Fraction of this source's distinct clusters to keep each epoch."),
    )
    cluster_column: str = Field(
        description=(
            "Name of the column containing cluster IDs to use for this source."
        ),
    )
    split_column: str | None = Field(
        None,
        description=(
            "Column used for the train/val holdout, defaulting to "
            "`cluster_column`. Pin it to a fixed coarse clustering so the "
            "holdout stays put when `cluster_column` changes, keeping the "
            "split+flatten cache valid and val sets comparable across "
            "thresholds. Assumes clusterings are nested."
        ),
    )
    extra_cluster_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Extra cluster columns to retain in the cache. List every threshold "
            "you intend to train on, so switching `cluster_column` rebuilds only "
            "the small cluster index instead of the split+flatten pass."
        ),
    )
    madvise_random: bool = Field(
        True,
        description=(
            "Faster and more efficient random row reads, at the expense of "
            "sequential read performance. Provides a madvise(MADV_RANDOM) hint "
            "to the Linux OS that read prefetching should be minimized. Enable "
            "when the dataset is too large to fit into the available system "
            "memory (page cache). Disable when the dataset easily fits in the "
            "system memory (page cache) or when it is read sequentially."
        ),
    )

    # `type: parquet` / `type: prepared` fields
    path: str | None = Field(
        None,
        description=(
            "Path, directory, or glob pattern to Parquet file(s) "
            "(type='parquet'), or an exported source folder (type='prepared'). "
            "Required for both."
        ),
    )

    # `type: hub` fields
    repo_id: str | None = Field(
        None,
        description=(
            'HF Hub repository ID, e.g. "owner/dataset-name". '
            "Required when type='hub'."
        ),
    )
    subset: str | None = Field(
        None, description="Dataset configuration/subset name (optional, type='hub')."
    )
    split: str = Field("train", description="Which split to load (type='hub').")
    revision: str | None = Field(
        None,
        description="Branch, tag, or commit hash to pin the dataset version (type='hub').",
    )

    @model_validator(mode="after")
    def _check_required_for_type(self) -> "DataSourceConfig":
        """Require ``path`` or ``repo_id`` depending on ``type``."""
        if self.type in ("parquet", "prepared") and not self.path:
            raise ValueError(f"`path` is required when type='{self.type}'.")
        if self.type == "hub" and not self.repo_id:
            raise ValueError("`repo_id` is required when type='hub'.")
        return self


class CurriculumConfig(BaseModel):
    """Training score threshold schedule (ramps from start_score to end_score, then holds)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["linear", "cosine", "exponential"] = "linear"
    start_score: float = Field(..., ge=0.0, le=1.0)
    end_score: float = Field(..., ge=0.0, le=1.0)
    ramp_epochs: int = Field(..., gt=0)

    @model_validator(mode="after")
    def _check_exponential_start(self) -> "CurriculumConfig":
        """Reject ``start_score=0`` for the exponential schedule."""
        if self.type == "exponential" and self.start_score <= 0.0:
            raise ValueError(
                "CurriculumConfig.start_score must be > 0 when type='exponential' "
                "(geometric interpolation start*(end/start)**p is undefined at 0)."
            )
        return self


def curriculum_score_threshold(curriculum: CurriculumConfig, epoch: int) -> float:
    """Calculate the epoch's score threshold, plateauing at ``end_score`` once ``ramp_epochs`` is reached."""
    p = min(max(epoch / curriculum.ramp_epochs, 0.0), 1.0)
    start, end = curriculum.start_score, curriculum.end_score

    if curriculum.type == "linear":
        return start + (end - start) * p
    if curriculum.type == "cosine":
        return start + (end - start) * (1 - math.cos(math.pi * p)) / 2
    if curriculum.type == "exponential":
        return start * (end / start) ** p
    raise ValueError(f"Unknown curriculum type: {curriculum.type!r}")


class DatasetConfig(BaseModel):
    """Configuration for map-style dataset construction."""

    model_config = ConfigDict(extra="forbid")

    sources: dict[str, DataSourceConfig] = Field(
        {},
        description=(
            "Mapping of source names to their config (`type='parquet'` or `type='hub'`)."
        ),
    )
    score_column: str = Field(
        "score",
        description="Name of the column containing quality scores for filtering.",
    )
    score_curriculum: CurriculumConfig = Field(
        description=(
            "Curriculum schedule for the train split's score threshold: "
            "ramps per-epoch from start_score to end_score over ramp_epochs, "
            "then holds at end_score."
        ),
    )
    val_min_score: float = Field(
        0.5,
        description=(
            "Fixed minimum score threshold for filtering the (static) "
            "validation split. Unlike the train split, this never varies "
            "across epochs."
        ),
    )
    val_holdout_modulus: int = Field(
        20,
        description=(
            "Deterministic validation holdout: a cluster is held out when "
            "hash(cluster_id) % val_holdout_modulus == 0, yielding roughly "
            "1/val_holdout_modulus of all clusters as validation data."
        ),
    )
    seed: int = Field(
        42,
        description="Base random seed; combined with the epoch number to vary "
        "per-cluster sequence choice and per-source subsampling across epochs.",
    )
    num_proc: int | None = Field(
        None,
        description="Number of processes for the one-time, cached passes over "
        "each source: the initial parquet/hub load into Arrow and the "
        "train/val split. Not part of any cache key.",
    )
    cluster_index_cache_dir: Path | None = Field(
        None,
        description="Directory for cached cluster->row-index maps, built "
        "once per source and reused across runs. Defaults to a subdirectory "
        "of the HF datasets cache.",
    )
    split_writer_batch_size: int = Field(
        100_000,
        gt=0,
        description="Rows per Arrow record batch in the cached train/val "
        "splits. `datasets` defaults to 1000, which leaves a billion-row "
        "source with millions of batches; opening one then costs a metadata "
        "read per batch, minutes to hours on a network filesystem. Not part "
        "of the cache key: an existing cache keeps its original batch size.",
    )
    prepare_only: bool = Field(
        False,
        description="Build the dataset caches (split, flatten, cluster index) "
        "and exit before constructing the model or training. Lets a slow "
        "first-time cache build run once on cheap CPU-only hardware, so the "
        "real run starts warm.",
    )
    local_stage_dir: Path | None = Field(
        None,
        description="Node-local dir. (e.g., `$SLURM_TMPDIR`) to copy "
        "each source's sequence-only train/val splits into before training. "
        "Leave unset when HF_DATASETS_CACHE is already on local disk. "
        "Ignored with `prepare_only`.",
    )
    export_dir: Path | None = Field(
        None,
        description="With `prepare_only`, also write each source's train/val "
        "splits and cluster index to `export_dir/<source>/`. The folder is "
        "self-contained: copy it anywhere (e.g. another filesystem) and load it "
        "with `type: prepared`, `path: <folder>`, without rebuilding. Ignored "
        "without `prepare_only`.",
    )

    @model_validator(mode="after")
    def _warn_export_without_prepare_only(self) -> "DatasetConfig":
        if self.export_dir is not None and not self.prepare_only:
            logger.warning(
                "`dataset.export_dir` is set but `dataset.prepare_only` is false: "
                "nothing will be exported. Set `prepare_only=true` to export."
            )
        return self


# ---------------------------------------------------------------------------
# Cluster index: group rows by cluster, and pick one row per cluster
# ---------------------------------------------------------------------------


@dataclass
class ClusterIndex:
    """CSR-style index grouping dataset rows by cluster ID.

    * order: Row indices sorted by ``(cluster, score descending)``.
    * boundaries: Slice indices marking each cluster's span in ``order``.
    * scores: Row scores aligned to ``order``.
    * lengths: Row sequence lengths aligned to ``order``.
    * cache_dir: Cache directory of the index, also used for derived caches
      (``None`` if in-memory only).
    """

    order: np.ndarray
    boundaries: np.ndarray
    scores: np.ndarray
    lengths: np.ndarray
    cache_dir: Path | None = None

    @property
    def num_clusters(self) -> int:
        """Number of clusters in the index."""
        return max(len(self.boundaries) - 1, 0)


def _build_cluster_index(
    dataset: HFDataset, cluster_column: str, score_column: str
) -> ClusterIndex:
    """Group dataset rows by cluster ID (requires a pre-flattened dataset)."""
    num_rows = len(dataset)
    if num_rows == 0:
        return ClusterIndex(
            order=np.array([], dtype=np.int64),
            boundaries=np.array([0], dtype=np.int64),
            scores=np.array([], dtype=np.float64),
            lengths=np.array([], dtype=np.int64),
        )

    # Log the cluster index build time for reporting.
    start = time.monotonic()
    logger.info(f"Building cluster index over {num_rows:,} rows...")

    table = dataset.data.table

    # Check if the dataset is flattened by looking for nested StructTypes
    if any(pa_types.is_struct(t) for t in table.schema.types):
        logger.warning(
            "The dataset appears to contain nested fields (StructTypes). "
            "_build_cluster_index expects a pre-flattened dataset; "
            "failing to flatten may result in missing columns or unexpected behavior."
        )

    # Reorder with numpy rather than Arrow `take`, which is slow over the many
    # chunks `flatten_indices` produces. Cluster IDs are dictionary-encoded so
    # we sort integer codes, not one Python str per row (~40-60 GB at 2B rows);
    # codes are remapped to lexicographic order to match sorting raw strings.
    encoded = table.column(cluster_column).dictionary_encode()
    logger.info(
        f"[cluster index] read '{cluster_column}' in {time.monotonic() - start:.1f}s"
    )
    encoded = encoded.combine_chunks()  # unify the per-chunk dictionaries
    if isinstance(encoded, pa.ChunkedArray):  # older pyarrow keeps the wrapper
        encoded = encoded.chunk(0)
    uniques, codes = encoded.dictionary, encoded.indices

    # Extra slot is a null sentinel: ranks last, matching null string ordering.
    rank_dtype = np.int32 if len(uniques) < np.iinfo(np.int32).max else np.int64
    lex_rank = np.empty(len(uniques) + 1, dtype=rank_dtype)
    lex_rank[pc.sort_indices(uniques).to_numpy(zero_copy_only=False)] = np.arange(
        len(uniques)
    )
    lex_rank[len(uniques)] = len(uniques)
    cluster_codes = lex_rank[
        pc.fill_null(codes, len(uniques)).to_numpy(zero_copy_only=False)
    ]
    logger.info(
        f"[cluster index] encoded {len(uniques):,} cluster ids in "
        f"{time.monotonic() - start:.1f}s"
    )
    # Keep the narrow source dtypes: upcasting to 64-bit would double both
    # cached arraysfor no gain, and they are mmapped on every training run.
    scores = table.column(score_column).to_numpy(zero_copy_only=False)
    if scores.dtype == np.float16:
        scores = scores.astype(np.float32)
    # Signed, so downstream arithmetic on lengths can't wrap around.
    lengths = (
        table.column("sequence_length").to_numpy(zero_copy_only=False).astype(np.int32)
    )
    logger.info(
        f"[cluster index] read '{score_column}' and 'sequence_length' in "
        f"{time.monotonic() - start:.1f}s"
    )

    # Sort by (cluster ascending, score descending)
    order = (
        pc.sort_indices(
            pa.table({"cluster": pa.array(cluster_codes), "score": pa.array(scores)}),
            sort_keys=[("cluster", "ascending"), ("score", "descending")],
        )
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    logger.info(f"[cluster index] sort done in {time.monotonic() - start:.1f}s")

    sorted_clusters = cluster_codes[order]
    sorted_scores = scores[order]
    sorted_lengths = lengths[order]

    change_points = np.flatnonzero(sorted_clusters[1:] != sorted_clusters[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [len(sorted_clusters)])).astype(
        np.int64
    )
    logger.info(
        f"Cluster index built: {len(boundaries) - 1:,} clusters, "
        f"{num_rows:,} rows, {time.monotonic() - start:.1f}s total."
    )
    return ClusterIndex(
        order=order,
        boundaries=boundaries,
        scores=sorted_scores,
        lengths=sorted_lengths,
    )


def load_or_build_cluster_index(
    dataset: HFDataset, cluster_column: str, score_column: str, cache_dir: Path
) -> ClusterIndex:
    """Load the cluster index from cache, or build and cache it.

    Keyed by ``dataset._fingerprint``. ``order``, ``scores`` and ``lengths`` are
    saved as ``.npy`` files and mmap-loaded. The cached row count is checked on
    load; a mismatch triggers a rebuild.
    """
    digest = hashlib.md5(
        f"{dataset._fingerprint}-{cluster_column}-{score_column}".encode("utf-8")
    ).hexdigest()
    index_dir = cache_dir / f"cluster_index-{digest}"
    manifest_path = index_dir / "manifest.json"

    if manifest_path.exists():
        cached_rows = json.loads(manifest_path.read_text()).get("num_rows")
        if cached_rows == len(dataset):
            logger.info(f"Cluster index cache hit: {index_dir}")
            return ClusterIndex(
                order=np.load(index_dir / "order.npy", mmap_mode="r"),
                boundaries=np.load(index_dir / "boundaries.npy"),
                scores=np.load(index_dir / "scores.npy", mmap_mode="r"),
                lengths=np.load(index_dir / "lengths.npy", mmap_mode="r"),
                cache_dir=index_dir,
            )
        logger.warning(
            f"Cluster index cache is stale (cached {cached_rows} rows, dataset "
            f"has {len(dataset):,}); rebuilding: {index_dir}"
        )
    else:
        logger.info(f"Cluster index cache miss: {index_dir}")

    index = _build_cluster_index(dataset, cluster_column, score_column)
    index_dir.mkdir(parents=True, exist_ok=True)
    write_start = time.monotonic()
    np.save(index_dir / "order.npy", index.order)
    np.save(index_dir / "boundaries.npy", index.boundaries)
    np.save(index_dir / "scores.npy", index.scores)
    np.save(index_dir / "lengths.npy", index.lengths)
    written = (
        index.order.nbytes
        + index.boundaries.nbytes
        + index.scores.nbytes
        + index.lengths.nbytes
    )
    logger.info(
        f"Cluster index written: {Bytes(written)} to {index_dir} in "
        f"{time.monotonic() - write_start:.1f}s"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": dataset._fingerprint,
                "cluster_column": cluster_column,
                "score_column": score_column,
                "num_rows": len(index.order),
                "num_clusters": index.num_clusters,
            },
            indent=2,
        )
    )
    index.cache_dir = index_dir
    return index


def _canonicalize_threshold(threshold: float) -> str:
    """Return a stable string key for *threshold*, safe for use in a filename.

    Floats are unsafe as raw cache-key material (repr/precision can drift
    subtly across platforms/runs); formatting to fixed precision avoids that.
    """
    return f"{threshold:.6f}"


def compute_or_load_valid_counts(index: ClusterIndex, threshold: float) -> np.ndarray:
    """Per-cluster count of rows with ``score >= threshold`` (cached, mmap-loaded).

    Since ``index.order`` is score-sorted within each cluster, this is the length
    of each cluster's passing prefix. Not cached if ``index.cache_dir`` is unset.
    """
    if index.num_clusters == 0:
        return np.array([], dtype=np.int64)

    cache_path = None
    if index.cache_dir is not None:
        cache_path = (
            index.cache_dir / f"valid_count-{_canonicalize_threshold(threshold)}.npy"
        )
        if cache_path.exists():
            return np.load(cache_path, mmap_mode="r")

    mask = np.asarray(index.scores) >= threshold
    group_starts = index.boundaries[:-1]
    group_ends = index.boundaries[1:]
    # Accumulate in int64 via the ufunc's `dtype` rather than casting `mask`
    # first: the cast would allocate 8 bytes/row (15 GB on bfd) in every rank.
    valid_counts = np.add.reduceat(mask.view(np.uint8), group_starts, dtype=np.int64)
    valid_counts[group_ends == group_starts] = 0  # empty clusters, if any

    if cache_path is not None:
        # Multiple ranks may race to compute+write this on a cache miss;
        # since all ranks compute the same `valid_counts`, it's fine if
        # several write it. Write to a rank-unique temp file and
        # `os.replace` it into place (atomic on POSIX) so readers only
        # ever see a complete file, whichever writer lands last.
        tmp_path = cache_path.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp_path, valid_counts)
        os.replace(tmp_path, cache_path)
    return valid_counts


def pick_one_valid_row_per_cluster(
    index: ClusterIndex, valid_counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Pick a random row per cluster among its ``valid_counts`` top-scoring rows.

    Clusters with no valid rows are skipped.
    """
    if index.num_clusters == 0:
        return np.array([], dtype=np.int64)

    has_valid = valid_counts > 0
    group_starts = index.boundaries[:-1]

    # max(..., 1) prevents randint(0, 0) errors for empty clusters (discarded later)
    offsets = rng.integers(0, np.maximum(valid_counts, 1))

    # The chosen row is directly the `offset`-th highest-scoring row in its
    # cluster's span — no searchsorted needed, since valid rows are a prefix.
    rows = index.order[group_starts + offsets]
    return rows[has_valid]


# ---------------------------------------------------------------------------
# Phase 1: prepare sources (run once, at startup)
#
# Load, filter, split, and cluster-index each source. Cached to disk; never
# repeats per epoch.
# ---------------------------------------------------------------------------


def _cluster_hash(cluster_id: str | int) -> int:
    """Return a stable integer hash for *cluster_id* using MD5."""
    return int(hashlib.md5(str(cluster_id).encode("utf-8")).hexdigest(), 16)


def _cluster_split_filter(
    batch: dict, cluster_column: str, modulus: int, keep_val: bool
) -> list[bool]:
    """Batched filter predicate implementing the deterministic val holdout."""
    out = []
    for cid in batch[cluster_column]:
        is_val = cid is not None and _cluster_hash(cid) % modulus == 0
        out.append(is_val if keep_val else not is_val)
    return out


def _add_sequence_length(batch: dict) -> dict:
    """Batched map: derive ``sequence_length`` from ``sequence`` when missing."""
    return {"sequence_length": [len(s) for s in batch["sequence"]]}


@dataclass
class PreparedSource:
    """Result of Phase 1 for a single configured data source."""

    name: str
    sampling_fraction: float
    train_dataset: HFDataset
    train_index: ClusterIndex
    val_dataset: HFDataset
    val_index: ClusterIndex


def stage_dataset_locally(
    dataset: HFDataset, stage_prefix: Path, num_proc: int | None = None
) -> HFDataset:
    """Copy ``dataset`` to ``{stage_prefix}-{fingerprint}`` and return the copy.

    Row order is preserved, so the cluster index still applies. Written via a
    tmp dir and rename; an existing complete copy (e.g. from another rank) is reused.
    """
    target = stage_prefix.parent / f"{stage_prefix.name}-{dataset._fingerprint}"
    if target.exists():
        staged = load_from_disk(str(target))
        if len(staged) == len(dataset):
            logger.info(f"Using local staged copy: {target}")
            return staged
        logger.warning(
            f"Local staged copy is stale ({len(staged):,} rows, expected "
            f"{len(dataset):,}); re-staging: {target}"
        )
        shutil.rmtree(target)

    # Rows only: `dataset.data.nbytes` walks every record batch through the
    # network filesystem.
    logger.info(f"Staging {len(dataset):,} rows to local disk: {target}")
    start = time.monotonic()
    tmp = target.parent / f"{target.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    # An explicit `num_shards` skips `save_to_disk`'s own size estimate (the
    # same slow `nbytes` walk).
    dataset.save_to_disk(str(tmp), num_shards=8 * (num_proc or 1), num_proc=num_proc)
    tmp.rename(target)
    logger.info(f"Staged to {target} in {time.monotonic() - start:.1f}s")
    return load_from_disk(str(target))


def _resolve_parquet_data_files(path: str) -> str:
    """Resolve directory paths to a parquet glob pattern."""
    if Path(path).is_dir():
        return str(Path(path) / "*.parquet")
    return path


# Avoid `datasets`' automatic fingerprinting and select the args.
# we want to be part of it (to avoid unecessary cache rebuilds).
def _source_identity(source: DataSourceConfig) -> tuple:
    """Stable identity for a source's underlying files.

    Parquet: (name, size, mtime) of every matched file, so in-place edits are
    detected. Hub: ``(repo_id, subset, split, revision)``.
    """
    if source.type == "parquet":
        assert source.path is not None
        pattern = _resolve_parquet_data_files(source.path)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files matched pattern: {pattern}")
        return tuple(
            (Path(f).name, Path(f).stat().st_size, Path(f).stat().st_mtime_ns)
            for f in files
        )
    return (source.repo_id, source.subset, source.split, source.revision)


def _split_fingerprint(
    source: DataSourceConfig,
    dataset_cfg: "DatasetConfig",
    split_col: str,
    keep_val: bool,
) -> str:
    """Stable fingerprint for one `_split()` output (train or val).

    We exclude params that do not change the data itself (`num_proc`,
    `split_writer_batch_size`) so tuning them no longer invalidates the cache.
    """
    payload = {
        "source": _source_identity(source),
        "split_column": split_col,
        "val_holdout_modulus": dataset_cfg.val_holdout_modulus,
        "keep_val": keep_val,
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _load_and_split_source(
    name: str, source: DataSourceConfig, dataset_cfg: DatasetConfig
) -> tuple[HFDataset, HFDataset]:
    """Load a parquet/hub source and split it into train/val by cluster.

    The holdout uses ``split_column`` (defaulting to ``cluster_column``), and
    the projection carries ``extra_cluster_columns``; together these keep the
    expensive split+flatten cache valid when ``cluster_column`` changes.
    """
    load_start = time.monotonic()
    logger.info(f"Loading source '{name}' ({source.type})...")
    if source.type == "parquet":
        assert source.path is not None  # enforced by DataSourceConfig's validator
        dataset = load_dataset(
            "parquet",
            data_files=_resolve_parquet_data_files(source.path),
            split="train",
            streaming=False,
            num_proc=dataset_cfg.num_proc,
        )
    else:
        assert source.repo_id is not None  # enforced by DataSourceConfig's validator
        dataset = load_dataset(
            source.repo_id,
            source.subset,
            split=source.split,
            revision=source.revision,
            streaming=False,
            num_proc=dataset_cfg.num_proc,
        )
    logger.info(
        f"Loaded '{name}': {len(dataset):,} rows in "
        f"{time.monotonic() - load_start:.1f}s (fingerprint {dataset._fingerprint})"
    )

    cluster_col = source.cluster_column
    split_col = source.split_column or cluster_col
    score_col = dataset_cfg.score_column
    if cluster_col not in dataset.column_names:
        raise KeyError(f"Cluster column '{cluster_col}' not found in dataset '{name}'")
    if split_col not in dataset.column_names:
        raise KeyError(f"Split column '{split_col}' not found in dataset '{name}'")
    if score_col not in dataset.column_names:
        raise KeyError(f"Score column '{score_col}' not found in dataset '{name}'")
    for extra in source.extra_cluster_columns:
        if extra not in dataset.column_names:
            raise KeyError(
                f"Extra cluster column '{extra}' not found in dataset '{name}'"
            )

    # Sorted so the projection depends on the *set* of cluster columns, not on
    # which one is selected: changing cluster_column then leaves it unchanged
    # and the split+flatten cache still hits.
    cluster_cols = {cluster_col, split_col, *source.extra_cluster_columns}
    selected_columns = ["sequence", score_col, *sorted(cluster_cols)]
    if "sequence_length" in dataset.column_names:
        selected_columns.append("sequence_length")
    dataset = dataset.select_columns(selected_columns)
    logger.info(
        f"Projected '{name}' to {len(selected_columns)} columns: {selected_columns}"
    )

    if "sequence_length" not in dataset.column_names:
        logger.info(
            f"'sequence_length' missing from '{name}': computing it over "
            f"{len(dataset):,} rows (full pass)..."
        )
        dataset = dataset.map(
            _add_sequence_length, batched=True, num_proc=dataset_cfg.num_proc
        )

    split_kwargs = {
        "cluster_column": split_col,
        "modulus": dataset_cfg.val_holdout_modulus,
    }

    def _split(keep_val: bool, label: str) -> HFDataset:
        phase_start = time.monotonic()
        logger.info(
            f"Preparing '{name}' {label} split on '{split_col}' (filter + flatten; "
            "logs nothing further while reading a warm cache)..."
        )
        batch_size = dataset_cfg.split_writer_batch_size
        # Explicit `new_fingerprint=` on both steps.
        fp = _split_fingerprint(source, dataset_cfg, split_col, keep_val)
        # An identity `map` rather than `flatten_indices`: the latter forwards
        # `writer_batch_size` but not `batch_size`, so its output is locked to
        # `map`'s 1000-row default. Output is otherwise identical.
        out = dataset.filter(
            _cluster_split_filter,
            batched=True,
            num_proc=dataset_cfg.num_proc,
            fn_kwargs={**split_kwargs, "keep_val": keep_val},
            desc=f"Splitting '{name}' {label}",
            new_fingerprint=f"{fp}-filter",
        ).map(
            batched=True,
            batch_size=batch_size,
            writer_batch_size=batch_size,
            num_proc=dataset_cfg.num_proc,
            desc=f"Flattening '{name}' {label}",
            new_fingerprint=f"{fp}-map",
        )
        logger.info(
            f"'{name}' {label} split ready: {len(out):,} rows in "
            f"{time.monotonic() - phase_start:.1f}s (fingerprint {out._fingerprint})"
        )
        return out

    return _split(keep_val=False, label="train"), _split(keep_val=True, label="val")


def _load_prepared_source(
    name: str, source: DataSourceConfig, dataset_cfg: DatasetConfig
) -> tuple[HFDataset, HFDataset]:
    """Load the train/val splits of a folder written by ``export_prepared_source``.

    ``load_from_disk`` assigns a fresh fingerprint to a multi-shard dataset,
    so the original is restored from ``meta.json``: the cluster index copied
    alongside is keyed by it.
    """
    assert source.path is not None  # enforced by DataSourceConfig's validator
    root = Path(source.path)
    meta = json.loads((root / "meta.json").read_text())
    # Frozen at export: a changed value here would otherwise be silently ignored.
    for key, requested in (
        ("split_column", source.split_column),
        ("val_holdout_modulus", dataset_cfg.val_holdout_modulus),
    ):
        if requested is not None and requested != meta[key]:
            raise ValueError(
                f"Exported source '{name}' ({root}) was built with {key}="
                f"{meta[key]!r}, but {requested!r} is configured. Re-export "
                f"from parquet/hub to change it, or set it back to {meta[key]!r}."
            )
    fingerprints = meta["fingerprints"]
    train_dataset = load_from_disk(str(root / "train"))
    val_dataset = load_from_disk(str(root / "val"))
    train_dataset._fingerprint = fingerprints["train"]
    val_dataset._fingerprint = fingerprints["val"]
    for col in (source.cluster_column, dataset_cfg.score_column):
        if col not in train_dataset.column_names:
            raise KeyError(
                f"Column '{col}' not found in exported source '{name}' ({root}); "
                f"available: {train_dataset.column_names}"
            )
    logger.info(
        f"Loaded exported source '{name}' from {root}: {len(train_dataset):,} "
        f"train / {len(val_dataset):,} val rows"
    )
    return train_dataset, val_dataset


def export_prepared_source(
    out_dir: Path,
    source: DataSourceConfig,
    dataset_cfg: DatasetConfig,
    train_dataset: HFDataset,
    val_dataset: HFDataset,
    indexes: list[ClusterIndex],
) -> None:
    """Write a self-contained copy of a prepared source to ``out_dir``.

    Contains ``train/``, ``val/``, ``cluster_index/`` and a ``meta.json`` with the
    split fingerprints, provenance and the HF cache paths it was built from.
    Written via a tmp dir and rename, replacing any previous export.
    """
    hf_cache: dict[str, str | list[str]] = {
        "split_dirs": sorted(
            {
                str(Path(f["filename"]).parent)
                for ds in (train_dataset, val_dataset)
                for f in ds.cache_files
            }
        ),
        "cluster_index_dirs": [str(index.cache_dir) for index in indexes],
    }
    if source.type == "hub":
        assert source.repo_id is not None
        hub_dir = Path(HF_HUB_CACHE) / f"datasets--{source.repo_id.replace('/', '--')}"
        if hub_dir.exists():
            hf_cache["hub_download_dir"] = str(hub_dir)
    logger.info(f"Exporting to {out_dir}...")
    start = time.monotonic()
    tmp = out_dir.parent / f"{out_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    num_proc = dataset_cfg.num_proc
    for label, ds in (("train", train_dataset), ("val", val_dataset)):
        # Explicit `num_shards`: see `stage_dataset_locally`.
        num_shards = max(1, min(8 * (num_proc or 1), len(ds)))
        ds.save_to_disk(str(tmp / label), num_shards=num_shards, num_proc=num_proc)
    for index in indexes:
        assert index.cache_dir is not None
        shutil.copytree(index.cache_dir, tmp / "cluster_index" / index.cache_dir.name)
    (tmp / "meta.json").write_text(
        json.dumps(
            {
                "fingerprints": {
                    "train": train_dataset._fingerprint,
                    "val": val_dataset._fingerprint,
                },
                "source": source.model_dump(),
                "source_identity": _source_identity(source),
                "split_column": source.split_column or source.cluster_column,
                "val_holdout_modulus": dataset_cfg.val_holdout_modulus,
                "score_column": dataset_cfg.score_column,
                "columns": train_dataset.column_names,
                "hf_cache": hf_cache,
            },
            indent=2,
            default=str,
        )
    )
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    logger.info(
        f"Exported to {out_dir} in {time.monotonic() - start:.1f}s. HF cache "
        f"it was built from (safe to delete once all runs use `type: prepared`; "
        f"also listed in meta.json): {json.dumps(hf_cache, indent=2)}"
    )


def prepare_source(
    name: str, source: DataSourceConfig, dataset_cfg: DatasetConfig
) -> PreparedSource:
    """Phase 1: load, split by cluster, and index a single source (all cached).

    Score filtering happens later in ``build_epoch_plan``. A ``prepared`` source
    reads an exported folder instead of loading and splitting.
    """
    cluster_col = source.cluster_column
    score_col = dataset_cfg.score_column
    if source.type == "prepared":
        train_dataset, val_dataset = _load_prepared_source(name, source, dataset_cfg)
        assert source.path is not None
        cache_dir = Path(source.path) / "cluster_index"
    else:
        train_dataset, val_dataset = _load_and_split_source(name, source, dataset_cfg)
        cache_dir = dataset_cfg.cluster_index_cache_dir or (
            Path(str(HF_DATASETS_CACHE)) / "flair_plm_cluster_index"
        )
    logger.info(f"Cluster index for '{name}' train on '{cluster_col}':")
    train_index = load_or_build_cluster_index(
        train_dataset, cluster_col, score_col, cache_dir
    )
    logger.info(f"Cluster index for '{name}' val on '{cluster_col}':")
    val_index = load_or_build_cluster_index(
        val_dataset, cluster_col, score_col, cache_dir
    )

    if dataset_cfg.export_dir is not None and dataset_cfg.prepare_only:
        export_prepared_source(
            dataset_cfg.export_dir / name,
            source,
            dataset_cfg,
            train_dataset,
            val_dataset,
            [train_index, val_index],
        )

    # Training reads only `sequence`, but `Dataset.__getitem__` materializes
    # every column, so each row would fault in one buffer per carried cluster
    # column. Projecting here is a zero-copy schema change (no rebuild); the
    # index above already captured the cluster/score/length columns it needs.
    train_dataset = train_dataset.select_columns(["sequence"])
    val_dataset = val_dataset.select_columns(["sequence"])

    if dataset_cfg.local_stage_dir is not None and not dataset_cfg.prepare_only:
        train_dataset = stage_dataset_locally(
            train_dataset,
            dataset_cfg.local_stage_dir / f"{name}-train",
            dataset_cfg.num_proc,
        )
        val_dataset = stage_dataset_locally(
            val_dataset,
            dataset_cfg.local_stage_dir / f"{name}-val",
            dataset_cfg.num_proc,
        )

    # Datasets too large to fit entirely into pageable memory should advise the
    # OS to read them randomly to minimize prefetching. This helps reduce memory
    # pressure and reduce network read pressure during thrashing.
    if source.madvise_random and not dataset_cfg.prepare_only:
        if not advise_available():
            logger.warning(f"madvise is unavailable: skipped for '{name}'.")
        else:
            cache_files = train_dataset.cache_files + val_dataset.cache_files
            paths = (Path(f["filename"]) for f in cache_files)
            results = advise_paths(paths, MADV_RANDOM)
            for path, result in results.paths.items():
                if result.errno:
                    logger.warning(
                        f"madvise failed for {path}: {os.strerror(result.errno)}. "
                        "Dataloader may be slower."
                    )
            logger.info(
                f"Advised OS to random read {Bytes(results.num_bytes)} across "
                f"{results.num_files} shards of '{name}'."
            )

    return PreparedSource(
        name=name,
        sampling_fraction=source.sampling_fraction,
        train_dataset=train_dataset,
        train_index=train_index,
        val_dataset=val_dataset,
        val_index=val_index,
    )


def prepare_sources(dataset_cfg: DatasetConfig) -> list[PreparedSource]:
    """Phase 1 entrypoint: prepare every configured source, once, up front."""
    if not dataset_cfg.sources:
        raise ValueError("`sources` must contain at least one dataset source.")

    prepared = []
    for name, source in dataset_cfg.sources.items():
        if source.sampling_fraction <= 0.0:
            logger.info(
                f"Skipping source '{name}': `sampling_fraction` is 0, so it "
                "would contribute nothing to training. Its path is not loaded."
            )
            continue
        prepared.append(prepare_source(name, source, dataset_cfg))

    if not prepared:
        raise ValueError(
            "All configured sources have `sampling_fraction` 0; at least one "
            "source must have a positive `sampling_fraction`."
        )
    return prepared


# ---------------------------------------------------------------------------
# Phase 2: build epoch dataset (run once per epoch)
#
# Resample prepared sources into this epoch's train/val dataset. Only index
# arrays move here; sequence bytes load lazily when a DataLoader worker
# fetches a row.
# ---------------------------------------------------------------------------


def _stable_seed_component(name: str) -> int:
    """Return a stable, small positive int derived from *name* for RNG seeding."""
    return int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (2**31)


@dataclass
class EpochPlan:
    """An epoch's row selection as plain NumPy index arrays.

    * row_ids: Global indices into the ``ConcatDataset`` over ``prepared_sources``.
    * sequence_lengths: Aligned with ``row_ids``, used by ``TokenBatchSampler``.
    """

    row_ids: np.ndarray
    sequence_lengths: np.ndarray


def build_source_epoch_plan(
    index: ClusterIndex,
    valid_counts: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> EpochPlan:
    """Pick one representative row per eligible cluster, then subsample by ``fraction``.

    Row ids are local to this source (offset later by ``build_epoch_plan``).
    """
    eligible = np.flatnonzero(valid_counts > 0)
    if len(eligible) == 0:
        empty = np.array([], dtype=np.int64)
        return EpochPlan(row_ids=empty, sequence_lengths=empty)

    if fraction < 1.0:
        keep = max(1, round(len(eligible) * fraction))
        eligible = rng.choice(eligible, size=keep, replace=False)

    offsets = rng.integers(0, valid_counts[eligible])
    local_idx = index.boundaries[:-1][eligible] + offsets
    return EpochPlan(
        row_ids=index.order[local_idx], sequence_lengths=index.lengths[local_idx]
    )


def build_epoch_plan(
    prepared_sources: list[PreparedSource],
    epoch: int,
    seed: int,
    dataset_cfg: DatasetConfig,
    split: Literal["train", "val"] = "train",
) -> EpochPlan:
    """Phase 2 (NumPy-only): assemble this epoch's row selection across all sources.

    Filters by the curriculum threshold (train) or ``val_min_score`` (val), keeps
    one row per cluster, subsamples by ``sampling_fraction``, then concatenates and
    shuffles. Row ids are offset like ``ConcatDataset.cumulative_sizes``.
    """
    if split == "train":
        threshold = curriculum_score_threshold(dataset_cfg.score_curriculum, epoch)
    else:
        threshold = dataset_cfg.val_min_score

    row_id_chunks = []
    length_chunks = []
    cumulative_offset = 0
    for prepared in prepared_sources:
        dataset = prepared.train_dataset if split == "train" else prepared.val_dataset
        index = prepared.train_index if split == "train" else prepared.val_index
        if index.num_clusters > 0:
            source_seed = seed + epoch + _stable_seed_component(prepared.name)
            rng = np.random.default_rng(source_seed)
            valid_counts = compute_or_load_valid_counts(index, threshold)
            source_plan = build_source_epoch_plan(
                index, valid_counts, prepared.sampling_fraction, rng
            )
            if len(source_plan.row_ids) > 0:
                row_id_chunks.append(source_plan.row_ids + cumulative_offset)
                length_chunks.append(source_plan.sequence_lengths)
        cumulative_offset += len(dataset)

    if not row_id_chunks:
        raise ValueError(
            f"No data available for split={split!r} after filtering/sampling."
        )

    row_ids = np.concatenate(row_id_chunks)
    sequence_lengths = np.concatenate(length_chunks)

    perm = np.random.default_rng(seed + epoch).permutation(len(row_ids))
    return EpochPlan(row_ids=row_ids[perm], sequence_lengths=sequence_lengths[perm])
