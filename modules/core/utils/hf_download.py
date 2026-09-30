from __future__ import annotations

import importlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Literal


logger = logging.getLogger(__name__)

_HF_HUB_MODULE = importlib.import_module("huggingface_hub")
SNAPSHOT_DOWNLOAD: Any = getattr(_HF_HUB_MODULE, "snapshot_download")


def resolve_hf_token(token: str | None = None) -> str | None:
    """Resolve a Hugging Face token from explicit arg or standard env vars."""
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def download_from_hf(
    repo_id: str,
    local_dir: str | Path,
    repo_type: Literal["model", "dataset", "space"] = "dataset",
    revision: str | None = None,
    token: str | None = None,
    force_download: bool = False,
    overwrite: bool = False,
    local_files_only: bool = False,
    allow_patterns: list[str] | str | None = None,
    ignore_patterns: list[str] | str | None = None,
) -> Path:
    """Download repository files from HF Hub into a local directory."""
    resolved_dir = Path(local_dir).expanduser().resolve()

    if overwrite and resolved_dir.exists():
        logger.info(
            "Overwrite enabled; removing existing download target: %s", resolved_dir
        )
        shutil.rmtree(resolved_dir)

    resolved_dir.mkdir(parents=True, exist_ok=True)

    target_branch = revision if revision else "main"
    logger.info(
        "Downloading from HF repo %s (type=%s, revision=%s) to %s",
        repo_id,
        repo_type,
        target_branch,
        resolved_dir,
    )

    SNAPSHOT_DOWNLOAD(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(resolved_dir),
        token=resolve_hf_token(token),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_files_only=local_files_only,
        force_download=force_download or overwrite,
    )

    logger.info("Download completed: %s", resolved_dir)
    return resolved_dir
