"""ProteinGym suite resolution and assay loading."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from modules.core.utils.hf_download import download_from_hf, resolve_hf_token
from modules.evaluate.src.proteingym.config import ProteinGymRunConfig

logger = logging.getLogger(__name__)


def resolve_suite(config: ProteinGymRunConfig) -> Path:
    workspace = config.workspace
    assert workspace is not None and workspace.dataset_repo_id is not None
    source = Path(workspace.dataset_repo_id).expanduser()
    if source.is_dir():
        root = source
    elif source.is_absolute() or workspace.dataset_repo_id.startswith((".", "~")):
        raise FileNotFoundError(f"Local dataset suite not found: {source}")
    else:
        subset = workspace.dataset_config_name or workspace.dataset_name
        root = Path(
            snapshot_download(
                repo_id=workspace.dataset_repo_id,
                repo_type="dataset",
                revision=workspace.dataset_revision,
                allow_patterns=[f"{subset}/**"],
                force_download=config.force_download,
            )
        )
    data = root / (workspace.dataset_config_name or workspace.dataset_name)
    if not (data / "reference.parquet").is_file() or not (data / "data").is_dir():
        raise FileNotFoundError(
            f"Expected reference.parquet and data/ under {data}. workspace.dataset_repo_id must point to the suite root."
        )
    return data


def resolve_data_dir(
    data_dir: Path, data_repo_id: str | None, force_download: bool
) -> Path:
    """Prefer a local or downloaded HF snapshot, then fall back to ``data_dir``."""
    if data_repo_id is None:
        return data_dir
    if not force_download:
        try:
            return download_from_hf(
                repo_id=data_repo_id,
                local_dir=data_dir,
                repo_type="dataset",
                token=resolve_hf_token(),
                local_files_only=True,
            )
        except Exception as exc:
            logger.info("No local HF snapshot available for %s: %s", data_repo_id, exc)
    try:
        return download_from_hf(
            repo_id=data_repo_id,
            local_dir=data_dir,
            repo_type="dataset",
            token=resolve_hf_token(),
            force_download=force_download,
        )
    except Exception as exc:
        logger.warning(
            "Could not resolve HF dataset %s; falling back to %s: %s",
            data_repo_id,
            data_dir,
            exc,
        )
        return data_dir
