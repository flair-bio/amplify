#!/usr/bin/env python
"""Source TAPE ProteinNet contact prediction from local LMDB splits."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import ConfigDict

from modules.core.utils.hf_download import resolve_hf_token
from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.hf_io import upload_dataset_artifact
from modules.evaluate.src.dataset_sourcing.outputs import (
    write_hf_split_files,
    write_seqkit_stats,
    write_stats,
)
from modules.evaluate.src.dataset_sourcing.pipeline import load_runtime_config_from_argv
from modules.evaluate.src.dataset_sourcing.preprocess import create_split_subsets

logger = logging.getLogger(__name__)

PROTEINNET_URL = (
    "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/proteinnet.tar.gz"
)
PROTEINNET_LMDB_URL = "https://github.com/songlab-cal/tape#lmdb-data"
SPLIT_MAP = {
    "train": "train",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


class TapeProteinNetConfig(DatasetSourcingConfig):
    model_config = ConfigDict(extra="forbid")

    work_dir: Path = Path("datasets/tape_proteinnet_sourcing")
    repo_name_prefix: str = "tape-"
    repo_prefix: str = "tape"
    commit_message: str = "Add sourced TAPE ProteinNet contact prediction dataset"
    archive_url: str = PROTEINNET_URL
    archive_path: Path | None = None
    source_folder: Path | None = Path("downloaded_datasets")
    include_train_unfiltered: bool = False


def _download_archive(config: TapeProteinNetConfig, downloads_dir: Path) -> Path:
    if config.archive_path is not None:
        return config.archive_path
    downloads_dir.mkdir(parents=True, exist_ok=True)
    archive_path = downloads_dir / "proteinnet.tar.gz"
    if (
        archive_path.exists()
        and not config.force_download
        and _is_valid_archive(archive_path)
    ):
        return archive_path
    logger.info("Downloading %s to %s", config.archive_url, archive_path)
    temporary_path = archive_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(config.archive_url, temporary_path)
        if not _is_valid_archive(temporary_path):
            raise ValueError(
                f"Downloaded ProteinNet archive is invalid: {temporary_path}"
            )
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return archive_path


def _is_valid_archive(archive_path: Path) -> bool:
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".json"):
                    return True
    except (OSError, EOFError, tarfile.TarError):
        logger.warning("Ignoring invalid ProteinNet archive: %s", archive_path)
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
) -> tuple[str, Path | dict[str, Path]]:
    """Prefer pre-staged LMDB splits; fall back to explicit archive inputs."""
    if config.archive_path is not None:
        return "archive", config.archive_path
    if config.source_folder is not None and config.source_folder.exists():
        lmdb_splits = _find_local_lmdb_splits(config.source_folder)
        if lmdb_splits:
            return "lmdb", lmdb_splits
        local_archive = config.source_folder / "proteinnet.tar.gz"
        if local_archive.exists():
            return _archive_source_type(local_archive), local_archive
    return "archive", _download_archive(config, downloads_dir)


def _archive_source_type(archive_path: Path) -> str:
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            if any(
                Path(member.name).parts[-2:-1]
                and Path(member.name).parts[-2].endswith(".lmdb")
                for member in tar
            ):
                return "lmdb_archive"
    except (OSError, EOFError, tarfile.TarError):
        return "archive"
    return "archive"


def convert_lmdb_archive(
    archive_path: Path, extraction_dir: Path, include_train_unfiltered: bool = False
) -> dict[str, pl.DataFrame]:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extraction_dir, filter="data")
    return convert_lmdb(
        _find_local_lmdb_splits(extraction_dir),
        include_train_unfiltered=include_train_unfiltered,
    )


def _rows_from_lmdb(path: Path, split: str) -> list[dict[str, Any]]:
    import lmdb
    import pickle

    rows: list[dict[str, Any]] = []
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=path.is_dir())
    try:
        with env.begin() as txn:
            for _key, value in txn.cursor():
                raw = pickle.loads(value)
                if isinstance(raw, dict):
                    rows.append(_contact_record_from_raw(raw, split))
    finally:
        env.close()
    return rows


def convert_lmdb(
    lmdb_paths: dict[str, Path], include_train_unfiltered: bool = False
) -> dict[str, pl.DataFrame]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for split, path in lmdb_paths.items():
        if split == "train_unfiltered" and not include_train_unfiltered:
            continue
        rows[split] = _rows_from_lmdb(path, split)
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
    coordinates = raw.get("tertiary")
    valid_mask = raw.get("valid_mask")
    sequence = raw.get("primary")
    if not isinstance(sequence, str):
        raise ValueError("ProteinNet JSON record is missing string field 'primary'.")
    if coordinates is None:
        raise ValueError("ProteinNet JSON record is missing field 'tertiary'.")
    if valid_mask is None:
        valid_mask = [True] * len(sequence)

    # Vectorized pairwise distances: a pure-Python double loop is too slow for real ProteinNet proteins.
    coords = np.asarray(coordinates, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    length = len(sequence)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    # Raw, unfiltered pairs (incl. self/near-diagonal) to match the biomap-research sourcing
    # convention; separation filtering happens later in normalize_contact_record.
    upper_triangle = np.triu(np.ones((length, length), dtype=bool), k=0)
    contacts = upper_triangle & mask[:, None] & mask[None, :] & (distances < 8.0)
    targets = np.argwhere(contacts).tolist()
    return {
        "id": str(raw.get("id", raw.get("protein_id", ""))),
        "sequence": sequence,
        "targets": targets,
        "valid_positions": np.nonzero(mask)[0].tolist(),
        "split": split,
    }


def _record_from_json(line: str, split: str) -> dict[str, Any]:
    return _contact_record_from_raw(json.loads(line), split)


def convert_archive(
    archive_path: Path, include_train_unfiltered: bool = False
) -> dict[str, pl.DataFrame]:
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    if include_train_unfiltered:
        rows["train_unfiltered"] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stem = Path(member.name).stem
            source_split = stem.split(".")[0]
            if source_split == "train_unfiltered" and not include_train_unfiltered:
                continue
            split = (
                "train_unfiltered"
                if source_split == "train_unfiltered"
                else SPLIT_MAP.get(source_split)
            )
            if split is None:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            for line_bytes in handle:
                line = line_bytes.decode("utf-8").strip()
                if line:
                    rows[split].append(_record_from_json(line, split))
    missing = [split for split in ("train", "validation", "test") if not rows[split]]
    if missing:
        raise ValueError(
            f"ProteinNet archive did not contain required split rows: {missing}"
        )
    return {
        split: pl.from_dicts(split_rows)
        for split, split_rows in rows.items()
        if split_rows
    }


def write_proteinnet_card(
    output_dir: Path,
    config: TapeProteinNetConfig,
    split_frames: dict[str, pl.DataFrame],
    source_type: str,
) -> Path:
    rows = "\n".join(
        f"- `{split}`: {frame.height} rows"
        for split, frame in sorted(split_frames.items())
    )
    source_name = (
        "TAPE ProteinNet LMDB splits"
        if source_type == "lmdb"
        else "TAPE ProteinNet archive"
    )
    card = f"""# tape-proteinnet-contact-prediction

Sourced from {source_name}.

## Data files

Parquet files are stored under `data/` with Hugging Face split names. TAPE `valid` is published as `validation`; `train_unfiltered` is omitted by default and included only when explicitly configured.

## Contact definition

Positive contacts are upper-triangle residue pairs (including self-pairs) with valid coordinates and C-alpha Euclidean distance below 8 Angstroms. Rows store the raw, unfiltered pairs in `targets` plus residue indices with valid coordinates in `valid_positions`; sequence-separation filtering (>=6) is applied downstream by `normalize_contact_record`, matching the biomap-research sourcing convention.

## Splits

{rows}

## Provenance

- LMDB source: `{PROTEINNET_LMDB_URL}`
- Archive fallback: `{config.archive_url}`
- Dataset family: TAPE ProteinNet contact prediction
"""
    path = output_dir / "README.md"
    path.write_text(card, encoding="utf-8")
    return path


def run(config: TapeProteinNetConfig) -> Path:
    token = resolve_hf_token(config.token)
    dataset_name = "proteinnet_contact_prediction"
    downloads_dir = config.work_dir / "downloads"
    output_dir = config.work_dir / "outputs" / dataset_name
    if output_dir.exists() and config.overwrite_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_type, source = _resolve_source(config, downloads_dir)
    if source_type == "lmdb":
        if not isinstance(source, dict):
            raise TypeError("Expected LMDB split mapping for 'lmdb' source type.")
        split_frames = convert_lmdb(
            source, include_train_unfiltered=config.include_train_unfiltered
        )
    elif source_type == "lmdb_archive":
        if not isinstance(source, Path):
            raise TypeError("Expected archive path for 'lmdb_archive' source type.")
        split_frames = convert_lmdb_archive(
            source,
            config.work_dir / "tmp" / "proteinnet_lmdb",
            include_train_unfiltered=config.include_train_unfiltered,
        )
    else:
        if not isinstance(source, Path):
            raise TypeError("Expected archive path for archive source type.")
        split_frames = convert_archive(
            source, include_train_unfiltered=config.include_train_unfiltered
        )
    if config.create_split_subsets:
        split_frames = create_split_subsets(
            split_frames,
            seed=config.seed,
            identity_threshold=config.validation_cluster_identity_threshold,
            coverage_threshold=config.validation_cluster_coverage_threshold,
            num_threads=config.validation_cluster_num_threads,
        )
    write_hf_split_files(output_dir, split_frames)
    write_stats(output_dir, split_frames)
    write_seqkit_stats(
        output_dir, dataset_name, split_frames, threads=config.stats_threads
    )
    write_proteinnet_card(output_dir, config, split_frames, source_type)
    if config.has_upload_target():
        upload_dataset_artifact(output_dir, dataset_name, config, token)
    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    config = load_runtime_config_from_argv(sys.argv[1:], model_cls=TapeProteinNetConfig)
    run(config)


if __name__ == "__main__":
    main()
