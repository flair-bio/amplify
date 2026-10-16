"""Hugging Face Hub download/upload glue for dataset-sourcing pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

from modules.core.utils.hf_download import download_from_hf
from modules.core.utils.hf_upload import upload_to_hf
from modules.evaluate.src.dataset_sourcing.config import (
    HubUploadConfig,
    join_repo_path,
)
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec

logger = logging.getLogger(__name__)


def download_dataset(
    dataset_name: str,
    downloads_dir: Path,
    token: str | None,
    force_download: bool,
    spec: DatasetSourceSpec,
    source_revision: str | None = None,
) -> Path:
    """Download one dataset snapshot and return its local data dir."""
    repo_id = spec.source_repo_id(dataset_name)
    local_dir = downloads_dir / dataset_name
    if source_revision:
        local_dir = local_dir / source_revision

    download_from_hf(
        repo_id=repo_id,
        local_dir=local_dir,
        repo_type="dataset",
        # Skip other repo assets (e.g. LFS-tracked raw archives) we don't need to process.
        allow_patterns=["data/**", "README*", "dataset_infos.json"],
        token=token,
        force_download=force_download,
        revision=source_revision,
    )

    data_dir = local_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected data directory missing: {data_dir}")
    return data_dir


def upload_dataset_artifact(
    source_path: Path,
    dataset_name: str,
    config: HubUploadConfig,
    token: str | None,
) -> None:
    """Upload one prepared dataset folder to the configured HF target, if any."""
    if not config.has_upload_target():
        logger.info(
            "No upload target configured. Set repo_owner (per_dataset_repo) or "
            "repo_id (single_repo). Skipping upload for %s.",
            dataset_name,
        )
        return
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

    artifact_patterns = [
        "data/**",
        "README.md",
        "label_vocabulary.json",
        "stats.json",
        "dataset_infos.json",
        "*.parquet",
    ]
    upload_to_hf(
        folder_path=source_path,
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=repo_subpath,
        token=token,
        revision=config.revision,
        commit_message=config.commit_message,
        private=config.private,
        allow_patterns=artifact_patterns,
        # Remove remote artifacts (e.g. dropped split views) absent locally; scoped to path_in_repo.
        delete_patterns=artifact_patterns,
    )
