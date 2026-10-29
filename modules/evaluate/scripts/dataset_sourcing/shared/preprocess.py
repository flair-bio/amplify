"""Load raw splits and normalize them into HF-friendly id/sequence/targets/split schemas."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import numpy as np
import polars as pl

from modules.data.src.steps.cluster import (
    MmseqsClusteringConfig,
    MmseqsCommandConfig,
    mmseqs_clustering,
)
from modules.evaluate.scripts.dataset_sourcing.shared.config import (
    DatasetSourcingConfig,
    UniProtDatasetSourcingConfig,
)

logger = logging.getLogger(__name__)

# Linclust is insensitive below ~50% identity; the cascaded workflow finds remote homologs.
SOURCING_MMSEQS_CONFIG = MmseqsClusteringConfig(
    command=MmseqsCommandConfig(workflow="easy-cluster")
)

_SPLIT_ALIASES: dict[str, list[str]] = {
    "train": ["train"],
    "val": ["valid", "validation", "val"],
    "test": ["test"],
}
CANONICAL_SPLITS = ("train", "validation", "test")
ValidationSplitMethod = Literal["source", "mmseqs2", "random", "empty"]

# Numeric targets with more distinct values than this are quantile-binned for stratification.
_MAX_CATEGORICAL_TARGET_VALUES = 20
_STRATIFICATION_QUANTILE_BINS = 10
_SPLIT_FRACTION_TOLERANCE = 1e-6


def _find_split_files(data_dir: Path, aliases: list[str]) -> list[Path]:
    """Find every parquet shard for the first split alias that has any."""
    for alias in aliases:
        matches = sorted(data_dir.glob(f"{alias}-*.parquet"))
        if matches:
            return matches
    return []


# Above this fraction of train rows sampled into a synthetic validation
# split, warn: matching test's row count can silently gut the training set
# when test is a large slice of the overall dataset (e.g. a 40/40/20 split).
_LARGE_VALIDATION_FRACTION = 0.25


class _InsufficientClusterCountError(ValueError):
    """Raised when clusters cannot populate every requested split."""


def _cluster_records(
    records: Iterable[tuple[object, str]], config: DatasetSourcingConfig
) -> pl.DataFrame:
    """Cluster ``(id, sequence)`` records with MMseqs2 at the configured thresholds."""
    with TemporaryDirectory(prefix="mmseqs_split_") as tmp_dir:
        fasta_path = Path(tmp_dir) / "sequences.fasta"
        fasta_path.write_text(
            "".join(f">{record_id}\n{sequence}\n" for record_id, sequence in records)
        )
        return mmseqs_clustering(
            fasta_file=str(fasta_path),
            identity_threshold=config.cluster_identity_threshold,
            coverage_threshold=config.cluster_coverage_threshold,
            num_threads=config.cluster_num_threads,
            mmseqs_config=SOURCING_MMSEQS_CONFIG,
        )


def _order_clusters(sizes: dict[str, int], seed: int, descending: bool) -> list[str]:
    """Order clusters by size; seeded tie-breaks ignore MMseqs output order."""
    cluster_ids = sorted(sizes)
    tie_breaks = dict(
        zip(cluster_ids, np.random.default_rng(seed).random(len(cluster_ids)))
    )
    direction = -1 if descending else 1
    return sorted(
        cluster_ids,
        key=lambda cluster: (direction * sizes[cluster], tie_breaks[cluster]),
    )


def _ensure_ids(
    frame: pl.DataFrame, dataset_name: str, split_name: str
) -> pl.DataFrame:
    """Derive ``<dataset>_<split>_<row>`` ids unless the frame already has ``id``."""
    if "id" in frame.columns:
        return frame
    return frame.with_columns(
        pl.format(
            "{}_{}_{}",
            pl.lit(dataset_name),
            pl.lit(split_name),
            pl.int_range(pl.len()),
        ).alias("id")
    )


def _resolve_renamed_column(
    frame: pl.DataFrame, column_rename: dict[str, str] | None, name: str
) -> str | None:
    """Find the raw column that ``minimal_preprocess()`` will later rename to
    ``name``. Returns ``None`` if no such column can be resolved."""
    if name in frame.columns:
        return name
    for old, new in (column_rename or {}).items():
        if new == name and old in frame.columns:
            return old
    return None


# Dedup priority: evaluation splits keep a shared sequence; train loses it.
_DEDUP_SPLIT_PRIORITY = ("test", "val", "train")


def clean_source_splits(
    split_frames: dict[str, pl.DataFrame],
    max_sequence_length: int | None,
    column_rename: dict[str, str] | None = None,
) -> dict[str, pl.DataFrame]:
    """Clean resolved columns, then deduplicate available sequences.

    Resolved sequences are trimmed; nulls in resolved sequence/target columns
    and sequences beyond the optional length limit are removed. Shared
    sequences are kept in test, then validation, then train. Run this before
    validation synthesis so split size and test-cluster exclusion use the same
    cleaned rows.
    """
    ordered = sorted(
        split_frames,
        key=lambda name: (
            _DEDUP_SPLIT_PRIORITY.index(name)
            if name in _DEDUP_SPLIT_PRIORITY
            else len(_DEDUP_SPLIT_PRIORITY)
        ),
    )
    cleaned: dict[str, pl.DataFrame] = {}
    seen: set[str] = set()
    for split_name in ordered:
        frame = split_frames[split_name]
        sequence_column = _resolve_renamed_column(frame, column_rename, "sequence")
        if sequence_column is None:
            cleaned[split_name] = frame
            continue

        frame = frame.filter(pl.col(sequence_column).is_not_null()).with_columns(
            pl.col(sequence_column).cast(pl.Utf8).str.strip_chars()
        )
        target_column = _resolve_renamed_column(frame, column_rename, "targets")
        if target_column is not None:
            frame = frame.filter(pl.col(target_column).is_not_null())
        if max_sequence_length is not None:
            frame = frame.filter(
                pl.col(sequence_column).str.len_chars() <= max_sequence_length
            )
        frame = frame.unique(subset=sequence_column, keep="first", maintain_order=True)
        if seen:
            frame = frame.filter(~pl.col(sequence_column).is_in(list(seen)))
        seen.update(frame[sequence_column].to_list())
        cleaned[split_name] = frame
    return {split_name: cleaned[split_name] for split_name in split_frames}


def merge_label_sets_by_sequence(
    rows: Iterable[tuple[str, str, set[str]]],
) -> list[tuple[str, str, set[str]]]:
    """Collapse identical sequences, keeping the first id and the union of labels.

    Run before counting label support so duplicate entries cannot inflate it.
    """
    merged: dict[str, tuple[str, set[str]]] = {}
    for row_id, sequence, labels in rows:
        if sequence in merged:
            merged[sequence][1].update(labels)
        else:
            merged[sequence] = (row_id, set(labels))
    return [(row_id, sequence, labels) for sequence, (row_id, labels) in merged.items()]


def filter_labels_by_split_support(
    split_frames: dict[str, pl.DataFrame],
    label_count: int,
    min_support_by_split: dict[str, int],
) -> tuple[dict[str, pl.DataFrame], list[int]]:
    """Keep label positions meeting positive-support floors in each split.

    Rows left with no positive retained label are dropped.
    """
    supports: dict[str, np.ndarray] = {}
    for split_name in min_support_by_split:
        if split_name not in split_frames:
            raise ValueError(f"Required support split '{split_name}' is missing.")
        matrix = _multihot_matrix(split_frames[split_name], label_count)
        if matrix.shape[1] != label_count:
            raise ValueError(
                f"Split '{split_name}' has target width {matrix.shape[1]}, "
                f"expected {label_count}."
            )
        supports[split_name] = matrix.sum(axis=0)

    retained = [
        label_index
        for label_index in range(label_count)
        if all(
            supports[split_name][label_index] >= minimum
            for split_name, minimum in min_support_by_split.items()
        )
    ]
    if not retained:
        raise ValueError("No labels meet the required per-split positive support.")

    filtered_frames: dict[str, pl.DataFrame] = {}
    for split_name, frame in split_frames.items():
        matrix = _multihot_matrix(frame, label_count)[:, retained]
        frame = frame.with_columns(
            pl.Series("targets", matrix.tolist(), dtype=pl.List(pl.Int8))
        )
        if split_name in min_support_by_split:
            frame = frame.filter(pl.Series(np.any(matrix, axis=1).tolist()))
        filtered_frames[split_name] = frame
    return filtered_frames, retained


def _multihot_matrix(frame: pl.DataFrame, label_count: int) -> np.ndarray:
    if frame.is_empty():
        return np.zeros((0, label_count), dtype=np.int8)
    try:
        return np.array(frame["targets"].to_list(), dtype=np.int8)
    except ValueError as error:
        raise ValueError(
            f"Targets must be fixed-width multi-hot vectors of width {label_count}."
        ) from error


def _cluster_validation_row_indices(
    indexed_train: pl.DataFrame,
    test_frame: pl.DataFrame,
    sequence_column: str,
    validation_size: int,
    config: DatasetSourcingConfig,
) -> pl.Series | None:
    """Return the ``_split_row_idx`` values to hold out for validation,
    chosen along MMseqs2 cluster boundaries from train clusters that share no
    cluster with test. Returns ``None`` if MMseqs2 is unavailable or returns no
    clusters, so the caller can fall back to random sampling.
    """
    fallback = (
        "falling back to random row sampling for the synthesized validation split, "
        "which does not separate homologs from train."
    )
    if shutil.which("mmseqs") is None:
        logger.warning("mmseqs not found on PATH; %s", fallback)
        return None

    records: list[tuple[object, str]] = list(
        indexed_train.select("_split_row_idx", sequence_column).iter_rows()
    )
    if sequence_column in test_frame.columns:
        records += [
            (f"test_{row_idx}", sequence)
            for row_idx, sequence in enumerate(test_frame[sequence_column])
        ]
    assignments = _cluster_records(records, config)
    if assignments.is_empty():
        logger.warning("MMseqs2 returned no clusters; %s", fallback)
        return None

    test_clusters = assignments.filter(pl.col("sequence_id").str.starts_with("test_"))[
        "cluster_id"
    ].to_list()
    # Train rows homologous to test stay in train but are not used for validation.
    assignments = assignments.filter(
        ~pl.col("sequence_id").str.starts_with("test_")
        & ~pl.col("cluster_id").is_in(test_clusters)
    ).with_columns(pl.col("sequence_id").cast(pl.UInt32).alias("_split_row_idx"))
    if assignments.height < validation_size:
        raise ValueError(
            "Too few train sequences remain after excluding clusters shared with test "
            "to synthesize validation."
        )
    cluster_sizes = assignments.group_by("cluster_id").agg(pl.len().alias("_size"))
    sizes_by_cluster = dict(zip(cluster_sizes["cluster_id"], cluster_sizes["_size"]))

    # Smallest clusters first so accumulation lands close to validation_size instead of
    # overshooting on one large cluster.
    selected: list[str] = []
    running_total = 0
    for cluster_id in _order_clusters(sizes_by_cluster, config.seed, descending=False):
        if running_total >= validation_size:
            break
        selected.append(cluster_id)
        running_total += sizes_by_cluster[cluster_id]

    logger.info(
        "Clustered validation split at %.0f%% sequence identity: %d rows "
        "across %d clusters (target was %d rows).",
        config.cluster_identity_threshold * 100,
        running_total,
        len(selected),
        validation_size,
    )

    return assignments.filter(pl.col("cluster_id").is_in(selected))["_split_row_idx"]


def create_validation_split_from_train(
    train_frame: pl.DataFrame,
    test_frame: pl.DataFrame,
    config: DatasetSourcingConfig,
    sequence_column: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, ValidationSplitMethod]:
    """Sample a validation split from train with size matched to test.

    When *sequence_column* names a real column in *train_frame* and MMseqs2
    is installed, validation is built from whole train clusters that share no
    cluster with test; train rows homologous to test remain in train. Falls
    back to uniform random row sampling otherwise, which matches the test row
    count exactly (whole clusters can only approximate it).
    """
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
        validation_row_idx = _cluster_validation_row_indices(
            indexed_train,
            test_frame,
            sequence_column,
            validation_size,
            config,
        )

    if validation_row_idx is not None:
        validation_frame = indexed_train.filter(
            pl.col("_split_row_idx").is_in(validation_row_idx.to_list())
        ).sort("_split_row_idx")
    else:
        validation_frame = indexed_train.sample(
            n=validation_size,
            shuffle=True,
            seed=config.seed,
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


def read_source_splits(data_dir: Path) -> dict[str, pl.DataFrame]:
    """Read the raw train/test (and optional val) parquet shards as published."""
    split_frames: dict[str, pl.DataFrame] = {}
    for normalized_split, aliases in _SPLIT_ALIASES.items():
        split_files = _find_split_files(data_dir, aliases)
        if not split_files:
            if normalized_split == "val":
                continue
            raise FileNotFoundError(
                f"Could not find required split {normalized_split} in {data_dir}"
            )
        split_frames[normalized_split] = pl.concat(
            [pl.read_parquet(path) for path in split_files]
        )
    return split_frames


def load_splits(
    data_dir: Path,
    config: DatasetSourcingConfig,
    column_rename: dict[str, str] | None = None,
) -> tuple[dict[str, pl.DataFrame], ValidationSplitMethod]:
    """Load and clean train/val/test frames, synthesizing val from train when absent.

    Rows are cleaned and deduplicated (see ``clean_source_splits``) before any
    validation split is synthesized. *column_rename* (the mapping
    ``minimal_preprocess()`` later applies) resolves the raw sequence column
    so a synthesized validation split can follow sequence clusters.
    """
    split_frames = clean_source_splits(
        read_source_splits(data_dir), config.max_sequence_length, column_rename
    )

    validation_method: ValidationSplitMethod = "source"
    if "val" not in split_frames:
        logger.warning(
            "Validation split not found under %s. Sampling validation rows from train using seed=%s.",
            data_dir,
            config.seed,
        )
        train_frame, validation_frame, validation_method = (
            create_validation_split_from_train(
                train_frame=split_frames["train"],
                test_frame=split_frames["test"],
                config=config,
                sequence_column=_resolve_renamed_column(
                    split_frames["train"], column_rename, "sequence"
                ),
            )
        )
        split_frames["train"] = train_frame
        split_frames["val"] = validation_frame

    return split_frames, validation_method


def minimal_preprocess(
    dataset_name: str,
    split_frames: dict[str, pl.DataFrame],
    column_rename: dict[str, str],
) -> dict[str, pl.DataFrame]:
    """Rename raw columns into id/sequence/targets/split HF-friendly frames.

    Expects already-cleaned rows (see ``clean_source_splits``). Existing ``id``
    values are kept; otherwise ids are derived as ``<dataset>_<split>_<row>``.
    """
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
        frame = _ensure_ids(frame, dataset_name, normalized_split)

        processed[normalized_split] = frame.select(
            ["id", "sequence", "targets", "split"]
        )

    if not processed:
        raise ValueError(
            f"No split frames found after preprocessing for {dataset_name}"
        )
    return processed


class _UnstratifiableTargetsError(ValueError):
    """Raised when targets have no meaningful per-row stratum."""


def target_strata(targets: list[object], multilabel: bool = False) -> list[str]:
    """Build strata for scalar, continuous, token-level, and multi-hot targets.

    Multi-hot rows use their globally rarest positive label, a cheap
    approximation of iterative stratification.
    """
    if multilabel:
        matrix = np.array(targets, dtype=np.int8)
        label_frequency = matrix.sum(axis=0)
        strata: list[str] = []
        for row in matrix:
            positives = np.flatnonzero(row)
            strata.append(
                f"rarest:{positives[np.argmin(label_frequency[positives])]}"
                if positives.size
                else "rarest:none"
            )
        return strata

    if targets and isinstance(targets[0], (list, tuple, np.ndarray)):
        strata = []
        for target in targets:
            if not isinstance(target, (list, tuple, np.ndarray)):
                raise TypeError(
                    "Cannot stratify a mixture of scalar and token targets."
                )
            token_values = cast(list[object], np.asarray(target, dtype=object).tolist())
            if token_values and isinstance(token_values[0], list):
                raise _UnstratifiableTargetsError(
                    "Pairwise targets (e.g. contacts) have no per-row stratum."
                )
            counts: dict[str, int] = {}
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
        if unique_count > min(_MAX_CATEGORICAL_TARGET_VALUES, max(2, len(values) // 2)):
            bin_count = min(_STRATIFICATION_QUANTILE_BINS, len(values))
            # Rank-based bins stay equal-sized even when many values tie.
            order = np.argsort(values, kind="stable")
            bins = np.empty(len(values), dtype=int)
            bins[order] = np.arange(len(values)) * bin_count // len(values)
            return [f"quantile:{value}" for value in bins]
    return [f"label:{value}" for value in targets]


def random_partition_indices(
    row_count: int, split_sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    """Shuffle row indices and slice them into consecutive splits of the given sizes.

    Sizes should sum to *row_count*; any shortfall is silently left unassigned.
    """
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
    """Greedily balance each stratum across seeded splits of the requested sizes."""
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
    config: DatasetSourcingConfig,
) -> dict[str, list[int]]:
    """Assign each complete MMseqs cluster to exactly one dataset split."""
    assignments = _cluster_records(enumerate(frame["sequence"]), config)

    clusters: dict[str, list[int]] = {}
    # Sort so seeded tie-breaks do not depend on MMseqs output order.
    for sequence_id, cluster_id in (
        assignments.select("sequence_id", "cluster_id")
        .with_columns(pl.col("sequence_id").cast(pl.Int64))
        .sort("cluster_id", "sequence_id")
        .iter_rows()
    ):
        clusters.setdefault(str(cluster_id), []).append(int(sequence_id))
    assigned_indices = {index for indices in clusters.values() for index in indices}
    if assigned_indices != set(range(frame.height)):
        raise ValueError("MMseqs clustering did not return every dataset row.")
    if len(clusters) < len(split_sizes):
        raise _InsufficientClusterCountError(
            f"Hold-cluster-out splitting requires at least {len(split_sizes)} "
            f"clusters, but MMseqs produced {len(clusters)}."
        )

    ordered_clusters = _order_clusters(
        {cluster: len(rows) for cluster, rows in clusters.items()},
        config.seed,
        descending=True,
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
    # Largest clusters first, each into the split furthest below its target share;
    # clusters are indivisible, so splits can overshoot their targets.
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
    cluster_config: DatasetSourcingConfig | None = None,
    stratification_column: str = "targets",
    multilabel: bool = False,
) -> dict[str, pl.DataFrame]:
    """Add pooled random, stratified, and hold-cluster-out benchmark splits.

    Canonical train/validation/test remain unchanged. Custom partitions use
    ``_<method>`` suffixes and pool all canonical rows before assignment.
    Stratification uses ``targets`` by default or the specified column; set
    *multilabel* for multi-hot targets. Stratified subsets are skipped with a
    warning when targets yield fewer than two strata or have no per-row
    stratum (e.g. contact pairs). The cluster strategy runs only when
    *cluster_config* supplies the MMseqs2 settings; it assigns whole clusters
    between splits and is skipped with a warning when there are too few
    clusters to populate all roles.
    """
    missing = [name for name in CANONICAL_SPLITS if name not in split_frames]
    if missing:
        raise ValueError(f"Cannot create split subsets without {missing}.")

    split_sizes = {name: split_frames[name].height for name in CANONICAL_SPLITS}
    empty = [name for name, size in split_sizes.items() if size == 0]
    if empty:
        raise ValueError(f"Cannot create split subsets with empty splits: {empty}.")
    combined = pl.concat([split_frames[name] for name in CANONICAL_SPLITS])
    if stratification_column not in combined.columns:
        raise ValueError(
            f"Cannot stratify split subsets: missing column {stratification_column!r}."
        )
    partitions = {
        "random": random_partition_indices(combined.height, split_sizes, seed),
    }
    try:
        strata = target_strata(
            combined[stratification_column].to_list(), multilabel=multilabel
        )
    except _UnstratifiableTargetsError as error:
        logger.warning("Skipping stratified split subsets: %s", error)
    else:
        if len(set(strata)) < 2:
            logger.warning(
                "Skipping stratified split subsets: targets yield a single stratum."
            )
        else:
            partitions["stratified"] = stratified_partition_indices(
                strata, split_sizes, seed
            )
    if cluster_config is not None:
        try:
            partitions["cluster"] = _hold_cluster_out_partition_indices(
                combined, split_sizes, cluster_config
            )
        except _InsufficientClusterCountError as error:
            logger.warning("Skipping optional cluster split subsets: %s", error)

    result = {
        role: split_frames[role].with_columns(pl.lit(role).alias("split"))
        for role in CANONICAL_SPLITS
    }
    for method, partition in partitions.items():
        for role, indices in partition.items():
            output_name = f"{role}_{method}"
            result[output_name] = combined[indices].with_columns(
                pl.lit(output_name).alias("split")
            )
    return result


def create_full_dataset_splits(
    frame: pl.DataFrame, config: UniProtDatasetSourcingConfig
) -> dict[str, pl.DataFrame]:
    """Partition a dataset with no pre-existing splits into train/validation/test
    (or any named split roles) along MMseqs2 cluster boundaries, reducing
    homology shared across splits at the configured identity/coverage.

    Unlike :func:`create_split_subsets`, which adds extra partitions *on top
    of* an already-split dataset, this builds the canonical splits themselves
    from a single unsplit frame -- for sources (e.g. a fresh UniProt query)
    that don't come with any train/validation/test assignment at all.
    """
    split_fractions = config.split_fractions
    if abs(sum(split_fractions.values()) - 1.0) > _SPLIT_FRACTION_TOLERANCE:
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
    empty = [name for name, size in split_sizes.items() if size <= 0]
    if empty:
        raise ValueError(
            f"{frame.height} rows are too few for split fractions {split_fractions}; "
            f"empty splits: {empty}."
        )

    partition = _hold_cluster_out_partition_indices(frame, split_sizes, config)
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
        frame = _ensure_ids(frame, dataset_name, normalized_split)

        normalized[normalized_split] = frame.select(["id", "sequence"])

    return normalized


def split_by_clusters(
    frame: pl.DataFrame,
    config: UniProtDatasetSourcingConfig,
    dataset_name: str,
    min_support_by_split: dict[str, int] | None = None,
    multilabel: bool = False,
) -> tuple[dict[str, pl.DataFrame], list[int] | None]:
    """Cluster-split an unsplit id/sequence/targets frame into output splits.

    With *min_support_by_split*, multi-hot label positions lacking support in
    any canonical split are dropped; the retained positions are returned.
    """
    split_frames = create_full_dataset_splits(frame, config)
    retained = None
    if min_support_by_split is not None:
        split_frames, retained = filter_labels_by_split_support(
            split_frames, len(frame["targets"][0]), min_support_by_split
        )
    split_frames = minimal_preprocess(dataset_name, split_frames, column_rename={})
    if config.create_split_subsets:
        split_frames = create_split_subsets(
            split_frames, config.seed, config, multilabel=multilabel
        )
    return split_frames, retained
