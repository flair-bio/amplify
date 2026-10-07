"""Load raw splits and normalize them into HF-friendly id/sequence/targets/split schemas."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import numpy as np
import polars as pl

from modules.data.src.steps.cluster import mmseqs_clustering

logger = logging.getLogger(__name__)

_SPLIT_ALIASES: dict[str, list[str]] = {
    "train": ["train"],
    "val": ["valid", "validation", "val"],
    "test": ["test"],
}
ValidationSplitMethod = Literal["source", "mmseqs2", "random", "empty"]


def find_split_file(data_dir: Path, aliases: list[str]) -> Path | None:
    """Find the first parquet shard matching any known split alias."""
    for alias in aliases:
        matches = sorted(data_dir.glob(f"{alias}-*.parquet"))
        if matches:
            return matches[0]
    return None


# Above this fraction of train rows sampled into a synthetic validation
# split, warn: matching test's row count can silently gut the training set
# when test is a large slice of the overall dataset (e.g. a 40/40/20 split).
_LARGE_VALIDATION_FRACTION = 0.25


class _InsufficientClusterCountError(ValueError):
    """Raised when clusters cannot populate every requested split."""


def _resolve_sequence_column(
    frame: pl.DataFrame, column_rename: dict[str, str] | None
) -> str | None:
    """Find the raw column that ``minimal_preprocess()`` will later rename to
    ``sequence``, so validation-split clustering can operate on actual
    sequence text before that rename happens. Returns ``None`` if no such
    column can be resolved."""
    if "sequence" in frame.columns:
        return "sequence"
    for old, new in (column_rename or {}).items():
        if new == "sequence" and old in frame.columns:
            return old
    return None


def filter_sequence_length(
    split_frames: dict[str, pl.DataFrame],
    max_sequence_length: int | None,
    column_rename: dict[str, str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Filter source frames before any split assignment or clustering."""
    if max_sequence_length is None:
        return split_frames

    filtered: dict[str, pl.DataFrame] = {}
    for split_name, frame in split_frames.items():
        sequence_column = _resolve_sequence_column(frame, column_rename)
        if sequence_column is None:
            filtered[split_name] = frame
            continue
        filtered[split_name] = frame.filter(
            pl.col(sequence_column).cast(pl.Utf8).str.strip_chars().str.len_chars()
            <= max_sequence_length
        )
    return filtered


def _cluster_validation_row_indices(
    indexed_train: pl.DataFrame,
    test_frame: pl.DataFrame,
    sequence_column: str,
    validation_size: int,
    seed: int,
    identity_threshold: float,
    coverage_threshold: float,
    num_threads: int,
) -> tuple[pl.Series, list[int]] | None:
    """Return the ``_split_row_idx`` values to hold out for validation,
    chosen along MMseqs2 sequence-identity cluster boundaries. Train rows in
    test clusters are excluded first. Returns ``None`` if
    MMseqs2 isn't installed, so the caller can fall back to random sampling.
    """
    if shutil.which("mmseqs") is None:
        logger.info(
            "mmseqs not found on PATH; falling back to random row sampling "
            "for the synthesized validation split."
        )
        return None

    with TemporaryDirectory(prefix="mmseqs_val_split_") as tmp_dir:
        fasta_path = Path(tmp_dir) / "train.fasta"
        with fasta_path.open("w") as fasta_file:
            for row_idx, sequence in indexed_train.select(
                "_split_row_idx", sequence_column
            ).iter_rows():
                fasta_file.write(f">{row_idx}\n{sequence}\n")
            if sequence_column in test_frame.columns:
                for row_idx, sequence in enumerate(test_frame[sequence_column]):
                    fasta_file.write(f">test_{row_idx}\n{sequence}\n")

        assignments = mmseqs_clustering(
            fasta_file=str(fasta_path),
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            num_threads=num_threads,
        )

    if assignments.is_empty():
        return None

    test_clusters = assignments.filter(pl.col("sequence_id").str.starts_with("test_"))[
        "cluster_id"
    ].to_list()
    assignments = assignments.filter(
        ~pl.col("sequence_id").str.starts_with("test_")
    ).with_columns(pl.col("sequence_id").cast(pl.UInt32).alias("_split_row_idx"))
    excluded = assignments.filter(pl.col("cluster_id").is_in(test_clusters))[
        "_split_row_idx"
    ].to_list()
    assignments = assignments.filter(~pl.col("cluster_id").is_in(test_clusters))
    if assignments.height < validation_size:
        raise ValueError(
            "Too few train sequences remain after excluding clusters shared with test "
            "to synthesize validation."
        )
    cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("_size"))
    sizes_by_cluster = dict(zip(cluster_sizes["cluster_id"], cluster_sizes["_size"]))

    # Smallest clusters first so the greedy accumulation below lands close to
    # validation_size instead of overshooting on one large cluster. Use a
    # seeded random secondary key so different seeds choose different clusters
    # among equal-sized ties without sacrificing the size ordering.
    cluster_ids = cluster_sizes["cluster_id"].to_list()
    rng = np.random.default_rng(seed)
    random_tie_break = dict(zip(cluster_ids, rng.random(len(cluster_ids))))
    cluster_ids.sort(
        key=lambda cluster_id: (
            sizes_by_cluster[cluster_id],
            random_tie_break[cluster_id],
        )
    )

    selected: list[str] = []
    running_total = 0
    for cluster_id in cluster_ids:
        if running_total >= validation_size:
            break
        selected.append(cluster_id)
        running_total += sizes_by_cluster[cluster_id]

    logger.info(
        "Clustered validation split at %.0f%% sequence identity: %d rows "
        "across %d clusters (target was %d rows).",
        identity_threshold * 100 if identity_threshold <= 1 else identity_threshold,
        running_total,
        len(selected),
        validation_size,
    )

    return assignments.filter(pl.col("cluster_id").is_in(selected))[
        "_split_row_idx"
    ], excluded


def create_validation_split_from_train(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    seed: int,
    sequence_column: str | None = None,
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
    num_threads: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample a validation split from train with size matched to test.

    When *sequence_column* names a real column in *train_frame* and MMseqs2
    is installed, train rows in upstream test clusters are excluded and
    validation rows are chosen along sequence-identity cluster boundaries
    (at *identity_threshold*/*coverage_threshold*). Falls back to uniform
    random row sampling otherwise,
    preserving the exact test row count unlike the clustered path above,
    which can only hold out whole clusters.
    """
    train, validation, _ = _create_validation_split_from_train_with_method(
        train_frame=train_frame,
        test_frame=test_frame,
        seed=seed,
        sequence_column=sequence_column,
        identity_threshold=identity_threshold,
        coverage_threshold=coverage_threshold,
        num_threads=num_threads,
    )
    return train, validation


def _create_validation_split_from_train_with_method(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    seed: int,
    sequence_column: str | None = None,
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
    num_threads: int = 1,
) -> tuple[pl.DataFrame, pl.DataFrame, ValidationSplitMethod]:
    validation_size = test_frame.height
    if validation_size == 0:
        return train_frame, train_frame.clear(), "empty"

    if train_frame.height < validation_size:
        raise ValueError(
            "Cannot synthesize validation split because train has fewer rows than test: "
            f"train={train_frame.height}, test={validation_size}"
        )

    validation_fraction = validation_size / train_frame.height
    if validation_fraction > _LARGE_VALIDATION_FRACTION:
        logger.warning(
            "Synthesized validation split will consume %.1f%% of train (%d of %d rows, "
            "matched to test's row count) leaving only %d rows for training. Consider "
            "capping validation size instead of matching test exactly.",
            validation_fraction * 100,
            validation_size,
            train_frame.height,
            train_frame.height - validation_size,
        )

    indexed_train = train_frame.with_row_index(name="_split_row_idx")

    validation_row_idx: pl.Series | None = None
    if sequence_column is not None and sequence_column in indexed_train.columns:
        clustered = _cluster_validation_row_indices(
            indexed_train,
            test_frame,
            sequence_column,
            validation_size,
            seed,
            identity_threshold,
            coverage_threshold,
            num_threads,
        )
        if clustered is not None:
            validation_row_idx, excluded = clustered
            indexed_train = indexed_train.filter(
                ~pl.col("_split_row_idx").is_in(excluded)
            )
            if len(excluded):
                logger.warning(
                    "Excluded %d train rows in test clusters.", len(excluded)
                )

    if validation_row_idx is not None:
        validation_frame = indexed_train.filter(
            pl.col("_split_row_idx").is_in(validation_row_idx.to_list())
        ).sort("_split_row_idx")
    else:
        validation_frame = indexed_train.sample(
            n=validation_size,
            shuffle=True,
            seed=seed,
        ).sort("_split_row_idx")
    # Anti-join on the row index removes exactly the sampled rows from train, leaving no overlap.
    remaining_train = indexed_train.join(
        validation_frame.select("_split_row_idx"),
        on="_split_row_idx",
        how="anti",
    ).sort("_split_row_idx")

    return (
        remaining_train.drop("_split_row_idx"),
        validation_frame.drop("_split_row_idx"),
        "mmseqs2" if validation_row_idx is not None else "random",
    )


def load_splits(
    data_dir: Path,
    seed: int,
    column_rename: dict[str, str] | None = None,
    validation_cluster_identity_threshold: float = 0.3,
    validation_cluster_coverage_threshold: float = 0.8,
    validation_cluster_num_threads: int = 1,
    max_sequence_length: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Load train/val/test frames, synthesizing val from train when absent."""
    split_frames, _ = load_splits_with_method(
        data_dir=data_dir,
        seed=seed,
        column_rename=column_rename,
        validation_cluster_identity_threshold=validation_cluster_identity_threshold,
        validation_cluster_coverage_threshold=validation_cluster_coverage_threshold,
        validation_cluster_num_threads=validation_cluster_num_threads,
        max_sequence_length=max_sequence_length,
    )
    return split_frames


def load_splits_with_method(
    data_dir: Path,
    seed: int,
    column_rename: dict[str, str] | None = None,
    validation_cluster_identity_threshold: float = 0.3,
    validation_cluster_coverage_threshold: float = 0.8,
    validation_cluster_num_threads: int = 1,
    max_sequence_length: int | None = None,
) -> tuple[dict[str, pl.DataFrame], ValidationSplitMethod]:
    """Load train/val/test frames, synthesizing val from train when absent.

    *column_rename* (the same mapping ``minimal_preprocess()`` later applies)
    is used, if given, to resolve which raw column holds sequences, so the
    synthesized validation split can be clustered by sequence identity (see
    ``create_validation_split_from_train``) instead of sampled purely at
    random.
    """
    split_frames: dict[str, pl.DataFrame] = {}
    for normalized_split, aliases in _SPLIT_ALIASES.items():
        split_file = find_split_file(data_dir, aliases)
        if split_file is None:
            if normalized_split == "val":
                continue
            raise FileNotFoundError(
                f"Could not find required split {normalized_split} in {data_dir}"
            )

        split_frames[normalized_split] = pl.read_parquet(split_file)

    split_frames = filter_sequence_length(
        split_frames, max_sequence_length, column_rename
    )

    validation_method: ValidationSplitMethod = "source"
    if "val" not in split_frames:
        logger.warning(
            "Validation split not found under %s. Sampling validation rows from train using seed=%s.",
            data_dir,
            seed,
        )
        train_frame, validation_frame, validation_method = (
            _create_validation_split_from_train_with_method(
                train_frame=split_frames["train"],
                test_frame=split_frames["test"],
                seed=seed,
                sequence_column=_resolve_sequence_column(
                    split_frames["train"], column_rename
                ),
                identity_threshold=validation_cluster_identity_threshold,
                coverage_threshold=validation_cluster_coverage_threshold,
                num_threads=validation_cluster_num_threads,
            )
        )
        split_frames["train"] = train_frame
        split_frames["val"] = validation_frame

    return split_frames, validation_method


def minimal_preprocess(
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
    max_sequence_length: int | None,
    column_rename: dict[str, str],
) -> dict[str, pl.DataFrame]:
    """Normalize raw splits into id/sequence/targets/split HF-friendly frames."""
    processed: dict[str, pl.DataFrame] = {}
    for split_name, frame in split_frames.items():
        normalized_split = "validation" if split_name == "val" else split_name

        rename_map = {
            old: new for old, new in column_rename.items() if old in frame.columns
        }
        if rename_map:
            frame = frame.rename(rename_map)

        required_cols = {"sequence", "targets"}
        missing = sorted(required_cols - set(frame.columns))
        if missing:
            raise ValueError(
                f"{dataset_name}/{split_name} missing required columns: {missing}. "
                f"Found columns: {frame.columns}"
            )

        frame = frame.with_columns(pl.lit(normalized_split).alias("split"))
        frame = frame.filter(
            pl.col("sequence").is_not_null() & pl.col("targets").is_not_null()
        )
        frame = frame.with_columns(pl.col("sequence").cast(pl.Utf8).str.strip_chars())

        if max_sequence_length is not None:
            frame = frame.filter(
                pl.col("sequence").str.len_chars() <= max_sequence_length
            )

        frame = frame.with_row_index(name="_row_idx")
        frame = frame.with_columns(
            # Upstream repos rarely ship stable ids, so derive one deterministically for traceability.
            pl.format(
                "{}_{}_{}",
                pl.lit(dataset_name),
                pl.lit(normalized_split),
                pl.col("_row_idx"),
            ).alias("id")
        ).drop("_row_idx")

        processed[normalized_split] = frame.select(
            ["id", "sequence", "targets", "split"]
        )

    if not processed:
        raise ValueError(
            f"No split frames found after preprocessing for {dataset_name}"
        )

    return processed


def target_strata(targets: list[object]) -> list[str]:
    """Build useful strata for scalar, continuous, and token-level targets."""
    if targets and isinstance(targets[0], (list, tuple, np.ndarray)):
        strata: list[str] = []
        for target in targets:
            if not isinstance(target, (list, tuple, np.ndarray)):
                raise TypeError(
                    "Cannot stratify a mixture of scalar and token targets."
                )
            counts: dict[str, int] = {}
            token_values = cast(list[object], np.asarray(target, dtype=object).tolist())
            for value in token_values:
                label = str(value)
                counts[label] = counts.get(label, 0) + 1
            dominant = (
                max(counts, key=lambda label: counts[label]) if counts else "empty"
            )
            strata.append(f"token:{dominant}")
        return strata

    values = np.asarray(targets)
    if np.issubdtype(values.dtype, np.number):
        unique_count = len(np.unique(values))
        if unique_count > min(20, max(2, len(values) // 2)):
            order = np.argsort(values, kind="stable")
            bins = np.empty(len(values), dtype=int)
            bins[order] = np.arange(len(values)) * min(10, len(values)) // len(values)
            return [f"quantile:{value}" for value in bins]
    return [f"label:{value}" for value in targets]


def random_partition_indices(
    row_count: int, split_sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    indices = np.random.default_rng(seed).permutation(row_count)
    result: dict[str, list[int]] = {}
    start = 0
    for split, size in split_sizes.items():
        result[split] = indices[start : start + size].tolist()
        start += size
    return result


def stratified_partition_indices(
    strata: list[str], split_sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {split: [] for split in split_sizes}
    assigned_by_stratum: dict[str, dict[str, int]] = {}
    indices_by_stratum: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        indices_by_stratum.setdefault(stratum, []).append(index)

    for stratum in sorted(indices_by_stratum):
        assigned_by_stratum[stratum] = {split: 0 for split in split_sizes}
        for index in rng.permutation(indices_by_stratum[stratum]):
            available = [
                split
                for split, target_size in split_sizes.items()
                if len(result[split]) < target_size
            ]
            split = min(
                available,
                key=lambda name: (
                    assigned_by_stratum[stratum][name] / split_sizes[name],
                    len(result[name]) / split_sizes[name],
                ),
            )
            result[split].append(int(index))
            assigned_by_stratum[stratum][split] += 1
    return result


def _hold_cluster_out_partition_indices(
    frame: pl.DataFrame,
    split_sizes: dict[str, int],
    seed: int,
    identity_threshold: float,
    coverage_threshold: float,
    num_threads: int,
) -> dict[str, list[int]]:
    """Assign each complete MMseqs cluster to exactly one dataset split."""
    with TemporaryDirectory(prefix="mmseqs_benchmark_splits_") as tmp_dir:
        fasta_path = Path(tmp_dir) / "all.fasta"
        with fasta_path.open("w") as fasta_file:
            for index, sequence in enumerate(frame["sequence"]):
                fasta_file.write(f">{index}\n{sequence}\n")
        assignments = mmseqs_clustering(
            fasta_file=str(fasta_path),
            identity_threshold=identity_threshold,
            coverage_threshold=coverage_threshold,
            num_threads=num_threads,
        )

    clusters: dict[str, list[int]] = {}
    for sequence_id, cluster_id in assignments.select(
        "sequence_id", "cluster_id"
    ).iter_rows():
        clusters.setdefault(str(cluster_id), []).append(int(sequence_id))
    assigned_indices = {index for indices in clusters.values() for index in indices}
    if assigned_indices != set(range(frame.height)):
        raise ValueError("MMseqs clustering did not return every dataset row.")
    if len(clusters) < len(split_sizes):
        raise _InsufficientClusterCountError(
            f"Hold-cluster-out splitting requires at least {len(split_sizes)} "
            f"clusters, but MMseqs produced {len(clusters)}."
        )

    rng = np.random.default_rng(seed)
    tie_breaks = {
        cluster: value for cluster, value in zip(clusters, rng.random(len(clusters)))
    }
    ordered_clusters = sorted(
        clusters, key=lambda cluster: (-len(clusters[cluster]), tie_breaks[cluster])
    )
    largest_cluster = len(clusters[ordered_clusters[0]])
    for split, target_size in split_sizes.items():
        if largest_cluster > target_size:
            logger.warning(
                "Largest MMseqs cluster has %d rows, exceeding the %s target of %d; "
                "cluster-held-out split sizes may overshoot targets.",
                largest_cluster,
                split,
                target_size,
            )
    result: dict[str, list[int]] = {split: [] for split in split_sizes}
    for cluster in ordered_clusters:
        split = max(
            split_sizes,
            key=lambda name: (
                (split_sizes[name] - len(result[name])) / split_sizes[name]
            ),
        )
        result[split].extend(clusters[cluster])
    return result


def create_split_subsets(
    split_frames: dict[str, pl.DataFrame],
    seed: int,
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
    num_threads: int = 1,
    include_cluster: bool = True,
    stratification_column: str = "targets",
) -> dict[str, pl.DataFrame]:
    """Add pooled random, stratified, and hold-cluster-out benchmark splits.

    Canonical train/validation/test remain unchanged. Custom partitions use
    ``_<method>`` suffixes and pool all canonical rows before assignment.
    Stratification uses ``targets`` by default or the specified scalar column.
    The cluster strategy assigns whole MMseqs clusters between splits and never
    stratifies rows within a cluster. Its subsets are skipped with a warning
    when there are too few clusters to populate all roles.
    """
    canonical_names = ("train", "validation", "test")
    missing = [name for name in canonical_names if name not in split_frames]
    if missing:
        raise ValueError(f"Cannot create split subsets without {missing}.")

    split_sizes = {name: split_frames[name].height for name in canonical_names}
    empty = [name for name, size in split_sizes.items() if size == 0]
    if empty:
        raise ValueError(f"Cannot create split subsets with empty splits: {empty}.")
    combined = pl.concat([split_frames[name] for name in canonical_names])
    if stratification_column not in combined.columns:
        raise ValueError(
            f"Cannot stratify split subsets: missing column {stratification_column!r}."
        )
    partitions = {
        "random": random_partition_indices(combined.height, split_sizes, seed),
        "stratified": stratified_partition_indices(
            target_strata(combined[stratification_column].to_list()),
            split_sizes,
            seed,
        ),
    }
    if include_cluster:
        try:
            partitions["cluster"] = _hold_cluster_out_partition_indices(
                combined,
                split_sizes,
                seed,
                identity_threshold,
                coverage_threshold,
                num_threads,
            )
        except _InsufficientClusterCountError as error:
            logger.warning("Skipping optional cluster split subsets: %s", error)

    result = {
        role: split_frames[role].with_columns(pl.lit(role).alias("split"))
        for role in canonical_names
    }
    for method, partition in partitions.items():
        for role, indices in partition.items():
            output_name = f"{role}_{method}"
            result[output_name] = combined[indices].with_columns(
                pl.lit(output_name).alias("split")
            )
    return result


def create_full_dataset_splits(
    frame: pl.DataFrame,
    split_fractions: dict[str, float],
    seed: int,
    identity_threshold: float = 0.3,
    coverage_threshold: float = 0.8,
    num_threads: int = 1,
) -> dict[str, pl.DataFrame]:
    """Partition a dataset with no pre-existing splits into train/validation/test
    (or any named split roles) along MMseqs2 cluster boundaries, so no split
    contains near-duplicates of another split's sequences.

    Unlike :func:`create_split_subsets`, which adds extra partitions *on top
    of* an already-split dataset, this builds the canonical splits themselves
    from a single unsplit frame -- for sources (e.g. a fresh UniProt query)
    that don't come with any train/validation/test assignment at all.
    """
    if abs(sum(split_fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must sum to 1.0, got {split_fractions}")

    split_sizes = {
        name: round(frame.height * fraction)
        for name, fraction in split_fractions.items()
    }
    # Rounding can leave the sizes off by a few rows from frame.height; put
    # the drift on the largest split rather than dropping/duplicating rows.
    drift = frame.height - sum(split_sizes.values())
    largest = max(split_sizes, key=lambda name: split_sizes[name])
    split_sizes[largest] += drift

    partition = _hold_cluster_out_partition_indices(
        frame, split_sizes, seed, identity_threshold, coverage_threshold, num_threads
    )
    return {
        name: frame[indices].with_columns(pl.lit(name).alias("split"))
        for name, indices in partition.items()
    }


def build_seqkit_frames_from_raw_splits(
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Normalize raw split frames to the id/sequence schema required for seqkit FASTA export."""
    normalized: dict[str, pl.DataFrame] = {}

    for split_name, frame in split_frames.items():
        normalized_split = "validation" if split_name == "val" else split_name

        # preprocess="none" skips minimal_preprocess()'s column_rename, so also accept raw 'seq'.
        if "sequence" not in frame.columns and "seq" in frame.columns:
            frame = frame.rename({"seq": "sequence"})

        if "sequence" not in frame.columns:
            raise ValueError(
                f"{dataset_name}/{split_name} is missing sequence column ('sequence' or 'seq'). "
                f"Found: {frame.columns}"
            )

        frame = frame.filter(pl.col("sequence").is_not_null())
        frame = frame.with_columns(pl.col("sequence").cast(pl.Utf8).str.strip_chars())

        if "id" not in frame.columns:
            frame = frame.with_row_index(name="_row_idx")
            frame = frame.with_columns(
                pl.format(
                    "{}_{}_{}",
                    pl.lit(dataset_name),
                    pl.lit(normalized_split),
                    pl.col("_row_idx"),
                ).alias("id")
            ).drop("_row_idx")

        normalized[normalized_split] = frame.select(["id", "sequence"])

    return normalized
