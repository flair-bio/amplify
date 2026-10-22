"""Convert staged or archived TAPE ProteinNet LMDB splits to contact datasets.

Unlike the HF-repository sources, this always builds the single fixed
``proteinnet_contact_prediction`` dataset. The ``lmdb`` package is imported
lazily and is not a project dependency; run with ``uv run --with lmdb``.
"""

from __future__ import annotations

import logging
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import ConfigDict

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.downloads import download_url
from modules.evaluate.src.dataset_sourcing.hf_io import upload_dataset_artifact
from modules.evaluate.src.dataset_sourcing.outputs import (
    generated_utc_date,
    write_dataset_stats,
    write_hf_split_files,
)
from modules.evaluate.src.dataset_sourcing.pipeline import (
    prepare_output_dir,
    run_script,
)
from modules.evaluate.src.dataset_sourcing.preprocess import create_split_subsets

logger = logging.getLogger(__name__)

PROTEINNET_LMDB_ARCHIVE_URL = (
    "http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/proteinnet.tar.gz"
)
PROTEINNET_LMDB_URL = "https://github.com/songlab-cal/tape#lmdb-data"
# Standard C-alpha contact cutoff.
CONTACT_DISTANCE_ANGSTROM = 8.0
# Map upstream split spellings to canonical Hugging Face split names.
SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


class TapeProteinNetConfig(DatasetSourcingConfig):
    """Config for TAPE ProteinNet; `datasets`/`all_datasets` are inherited but ignored."""

    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Path("datasets/tape_proteinnet_sourcing")
    repo_name_prefix: str = "tape-"
    repo_prefix: str = "tape"
    commit_message: str = "Add sourced TAPE ProteinNet contact prediction dataset"
    archive_url: str = PROTEINNET_LMDB_ARCHIVE_URL
    # Explicit local archive; takes precedence over source_folder and archive_url.
    archive_path: Path | None = None
    # Searched recursively for *.lmdb splits (or a proteinnet.tar.gz) before downloading.
    source_folder: Path | None = Path("downloaded_datasets")
    # Dropped when create_split_subsets is true: subset generation keeps only train/validation/test.
    include_train_unfiltered: bool = False


def _is_valid_archive(archive_path: Path) -> bool:
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            return any(
                any(part.endswith(".lmdb") for part in Path(member.name).parts)
                for member in tar
            )
    except (OSError, EOFError, tarfile.TarError):
        logger.warning("Ignoring invalid ProteinNet LMDB archive: %s", archive_path)
    return False


def _lmdb_split_from_filename(path: Path) -> str | None:
    stem = path.name[: -len(".lmdb")] if path.name.endswith(".lmdb") else path.stem
    if "train_unfiltered" in stem:
        return "train_unfiltered"
    return SPLIT_MAP.get(stem.rsplit("_", 1)[-1])


def _find_local_lmdb_splits(source_folder: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for candidate in sorted(source_folder.rglob("*.lmdb")):
        split = _lmdb_split_from_filename(candidate)
        if split:
            found[split] = candidate
    return found


def _resolve_source(
    config: TapeProteinNetConfig, downloads_dir: Path
) -> Path | dict[str, Path]:
    """Resolve LMDB inputs: `archive_path`, then staged LMDBs, then a staged archive, then download."""
    if config.archive_path is not None:
        return config.archive_path
    if config.source_folder is not None and config.source_folder.exists():
        lmdb_splits = _find_local_lmdb_splits(config.source_folder)
        if lmdb_splits:
            return lmdb_splits
        local_archive = config.source_folder / "proteinnet.tar.gz"
        if local_archive.exists():
            return local_archive
    return download_url(
        config.archive_url,
        downloads_dir / "proteinnet.tar.gz",
        force=config.force_download,
        validator=_is_valid_archive,
        cache_identity=config.archive_url,
    )


def convert_lmdb_archive(
    archive_path: Path,
    extraction_dir: Path,
    include_train_unfiltered: bool = False,
    max_sequence_length: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Extract recognized LMDB members, then convert the discovered splits."""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extraction_dir, filter="data")
    return convert_lmdb(
        _find_local_lmdb_splits(extraction_dir),
        include_train_unfiltered=include_train_unfiltered,
        max_sequence_length=max_sequence_length,
    )


def _rows_from_lmdb(
    path: Path, split: str, max_sequence_length: int | None = None
) -> list[dict[str, Any]]:
    """Read one LMDB split, filtering long proteins before pairwise distances."""
    import pickle

    import lmdb

    rows: list[dict[str, Any]] = []
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=path.is_dir())
    try:
        with env.begin() as txn:
            for _key, value in txn.cursor():
                # TAPE stores each LMDB record as a pickled JSON-like mapping.
                # Unpickling can execute arbitrary code: only read LMDBs from a trusted source.
                raw = pickle.loads(value)
                if not isinstance(raw, dict):
                    continue
                sequence = raw.get("primary")
                # Skip long proteins before building their quadratic contact map.
                if (
                    max_sequence_length is not None
                    and isinstance(sequence, str)
                    and len(sequence) > max_sequence_length
                ):
                    continue
                rows.append(_contact_record_from_raw(raw, split))
    finally:
        env.close()
    return rows


def convert_lmdb(
    lmdb_paths: dict[str, Path],
    include_train_unfiltered: bool = False,
    max_sequence_length: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Convert available LMDB splits and require nonempty canonical splits."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, path in lmdb_paths.items():
        if split == "train_unfiltered" and not include_train_unfiltered:
            continue
        rows[split] = _rows_from_lmdb(path, split, max_sequence_length)
    missing = [
        split for split in ("train", "validation", "test") if not rows.get(split)
    ]
    if missing:
        raise ValueError(
            f"ProteinNet LMDB source did not contain required split rows: {missing}"
        )
    return {
        split: pl.from_dicts(split_rows)
        for split, split_rows in rows.items()
        if split_rows
    }


def _contact_record_from_raw(raw: dict[str, Any], split: str) -> dict[str, Any]:
    """Build a raw contact record from TAPE coordinates and residue validity.

    Targets include valid upper-triangle C-alpha pairs below the distance
    cutoff, including self and near-diagonal pairs. Evaluation applies the
    minimum sequence separation and uses valid_positions to mask unknown pairs.
    """
    coordinates = raw.get("tertiary")
    valid_mask = raw.get("valid_mask")
    sequence = raw.get("primary")
    if not isinstance(sequence, str):
        raise TypeError("ProteinNet JSON record is missing string field 'primary'.")
    if coordinates is None:
        raise ValueError("ProteinNet JSON record is missing field 'tertiary'.")
    if valid_mask is None:
        # Treat all coordinates as usable when the source omits its residue mask.
        valid_mask = [True] * len(sequence)

    # Vectorized pairwise distances: a pure-Python double loop is too slow for real ProteinNet proteins.
    coords = np.asarray(coordinates, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    length = len(sequence)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # Keep raw pairs (including self and near-diagonal pairs); the evaluation
    # task applies its separation threshold after cropping.
    upper_triangle = np.triu(np.ones((length, length), dtype=bool), k=0)
    contacts = (
        upper_triangle
        & mask[:, None]
        & mask[None, :]
        & (distances < CONTACT_DISTANCE_ANGSTROM)
    )
    targets = np.argwhere(contacts).tolist()
    return {
        "id": str(raw.get("id", raw.get("protein_id", ""))),
        "sequence": sequence,
        "targets": targets,
        "valid_positions": np.nonzero(mask)[0].tolist(),
        "split": split,
    }


def write_proteinnet_card(
    output_dir: Path,
    config: TapeProteinNetConfig,
    split_frames: dict[str, pl.DataFrame],
    source: Path | dict[str, Path],
) -> Path:
    """Write the contact-label definition and source split handling to the card."""
    rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    if isinstance(source, dict):
        source_detail = "Local LMDB inputs: " + ", ".join(
            f"`{path.name}`" for path in source.values()
        )
    else:
        source_detail = (
            f"LMDB archive: `{source.name}` (download URL `{config.archive_url}`)"
        )
    subset_settings = (
        "Canonical splits are retained and pooled `_random` and `_cluster` "
        f"alternatives are generated with seed `{config.seed}`; `_stratified` is "
        "not generated because contact pairs have no per-row stratum. Cluster "
        f"identity is `{config.cluster_identity_threshold}`, coverage is "
        f"`{config.cluster_coverage_threshold}`, and threads is "
        f"`{config.cluster_num_threads}`."
        if config.create_split_subsets
        else "No alternative subsets are generated; upstream split assignments are retained."
    )
    if config.include_train_unfiltered and config.create_split_subsets:
        unfiltered_setting = (
            "Requested, but omitted from final output because alternate subset "
            "generation keeps only canonical split roles."
        )
    elif config.include_train_unfiltered:
        unfiltered_setting = "Included when present in the source."
    else:
        unfiltered_setting = "Omitted."
    card = f"""# tape-proteinnet-contact-prediction

Sourced from TAPE ProteinNet LMDB splits.

## Intended use

Protein residue-contact prediction from sequence. Contacts report C-alpha
proximity in the source structure; they are structural labels, not sequence-only
measurements.

## Data files

Parquet files are stored under `data/` with Hugging Face split names. TAPE `valid` is published as `validation`; `train_unfiltered` is omitted by default and included only when explicitly configured.

Dataset statistics are available in [`stats.json`](stats.json) at the dataset root.
The file includes row counts by split, columns, and SeqKit sequence-length metrics.

## Contact definition

Positive contacts are upper-triangle residue pairs (including self-pairs) with valid coordinates and C-alpha Euclidean distance below {CONTACT_DISTANCE_ANGSTROM:g} Angstroms. Rows store raw pairs in `targets` and resolved residue indices in `valid_positions`. Evaluation preserves unresolved pairs as unknown and applies the minimum sequence separation of 6 after cropping.

## Processing and splits

- {source_detail}
- `include_train_unfiltered`: `{str(config.include_train_unfiltered).lower()}`. {unfiltered_setting}
- `max_sequence_length`: `{config.max_sequence_length}`. Applied before optional subset generation and clustering.
- `create_split_subsets`: `{str(config.create_split_subsets).lower()}`. {subset_settings}
- Card generated (UTC): `{generated_utc_date()}`

Output row counts:

{rows}

## Provenance

- TAPE project and LMDB guidance: `{PROTEINNET_LMDB_URL}`
- Dataset family: TAPE ProteinNet contact prediction
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run(config: TapeProteinNetConfig) -> Path:
    """Convert configured LMDB splits, write artifacts, and optionally upload."""
    token = resolve_hf_token(config.token)
    dataset_name = "proteinnet_contact_prediction"
    downloads_dir = config.work_dir / "downloads"
    output_dir = prepare_output_dir(config, dataset_name)
    source = _resolve_source(config, downloads_dir)
    if isinstance(source, dict):
        split_frames = convert_lmdb(
            source,
            include_train_unfiltered=config.include_train_unfiltered,
            max_sequence_length=config.max_sequence_length,
        )
    else:
        split_frames = convert_lmdb_archive(
            source,
            config.work_dir / "tmp" / "proteinnet_lmdb",
            include_train_unfiltered=config.include_train_unfiltered,
            max_sequence_length=config.max_sequence_length,
        )
    if config.create_split_subsets:
        split_frames = create_split_subsets(split_frames, config.seed, config)
    write_hf_split_files(output_dir, split_frames)
    write_dataset_stats(output_dir, split_frames, threads=config.stats_threads)
    write_proteinnet_card(output_dir, config, split_frames, source)
    upload_dataset_artifact(output_dir, dataset_name, config, token)
    return output_dir


def main() -> None:
    run_script(TapeProteinNetConfig, run)


if __name__ == "__main__":
    main()
