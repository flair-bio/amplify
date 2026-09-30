"""Hugging Face Hub download/upload glue for dataset-sourcing pipelines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Protocol

from modules.core.utils.hf_download import download_from_hf
from modules.core.utils.hf_upload import upload_to_hf
from modules.evaluate.src.dataset_sourcing.config import (
    DatasetSourcingConfig,
    join_repo_path,
)
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec

logger = logging.getLogger(__name__)


class DatasetUploadConfig(Protocol):
    """Minimal configuration contract required by the shared uploader."""

    upload_layout: Literal["per_dataset_repo", "single_repo"]
    repo_prefix: str
    revision: str | None
    commit_message: str
    private: bool

    def resolve_upload_repo_id(self, dataset_name: str) -> str: ...


def download_dataset(
    dataset_name: str,
    downloads_dir: Path,
    token: str | None,
    force_download: bool,
    spec: DatasetSourceSpec,
) -> Path:
    """Download one dataset snapshot and return its local data dir."""
    repo_id = spec.source_repo_id(dataset_name)
    local_dir = downloads_dir / dataset_name

    download_from_hf(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        # Skip other repo assets (e.g. LFS-tracked raw archives) we don't need to process.
        allow_patterns=["data/**", "README*", "dataset_infos.json"],
        token=token,
        force_download=force_download,
    )

    data_dir = local_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory missing: {data_dir}")
    return data_dir


def upload_dataset_artifact(
    source_path: Path,
    dataset_name: str,
    config: DatasetUploadConfig,
    token: str | None,
) -> None:
    """Upload one prepared dataset folder to the configured HF target."""
    repo_id = config.resolve_upload_repo_id(dataset_name)

    # single_repo layout nests each dataset under repo_prefix/<dataset_name> in one repo;
    # per_dataset_repo already encodes the dataset name in the repo id itself.
    repo_subpath = (
        join_repo_path(config.repo_prefix, dataset_name)
        if config.upload_layout == "single_repo"
        else None
    )

    logger.info(
        "Uploading %s to datasets/%s/%s",
        source_path,
        repo_id,
        repo_subpath,
    )

    upload_to_hf(
        folder_path=source_path,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=repo_subpath,
        token=token,
        revision=config.revision,
        commit_message=config.commit_message,
        private=config.private,
        allow_patterns=[
            "data/**",
            "README.md",
            "stats.json",
            "stats/**",
            "dataset_infos.json",
            "reference.parquet",
            "combined.parquet",
        ],
    )
