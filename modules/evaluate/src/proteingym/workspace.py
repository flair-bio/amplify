"""Shared workspace lifecycle for the dedicated ProteinGym entry points."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from accelerate import PartialState
from huggingface_hub import snapshot_download

from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.proteingym.config import ProteinGymRunConfig
from modules.evaluate.src.utils.io import log_to_file, update_run_manifest

logger = logging.getLogger(__name__)


def _resolve_suite(config: ProteinGymRunConfig) -> Path:
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
            f"Expected reference.parquet and data/ under {data}. "
            "workspace.dataset_repo_id must point to the suite root."
        )
    return data


@contextmanager
def protein_gym_run[RunConfig: ProteinGymRunConfig](
    config: RunConfig, *, overwrite: bool = False
) -> Iterator[RunConfig]:
    """Reserve shared run paths before loading models; keep legacy paths intact."""
    if PartialState().num_processes != 1:
        raise ValueError("ProteinGym runners require --num_processes 1")
    if config.workspace is None:
        yield config
        return
    workspace = get_evaluation_workspace(
        config.workspace,
        variant=config.run_variant(config.model_dump()),
        seed=config.seed,
    )
    with workspace.reserve_run(overwrite=overwrite) as targets:
        resolved = config.model_copy(update={"data_dir": _resolve_suite(config)})
        settings = resolved.model_dump(
            mode="json",
            exclude={"model_id", "data_dir", "data_repo_id", "output_dir"},
        )
        inputs = {"model_id": resolved.model_id, "data_dir": str(resolved.data_dir)}
        workspace.validate_experiment("inputs", inputs)
        workspace.initialize_run(targets, overwrite=overwrite, config=settings)
        update_run_manifest(workspace.run_manifest_path, "inputs", inputs)
        logger.info("ProteinGym workspace: %s", workspace.run_root)
        with log_to_file(workspace.run_root / "log"):
            try:
                yield resolved
            except BaseException as error:
                update_run_manifest(
                    workspace.run_manifest_path,
                    "evaluation",
                    {"status": "failed", "config": settings, "error": str(error)},
                )
                raise
            else:
                update_run_manifest(
                    workspace.run_manifest_path,
                    "evaluation",
                    {"status": "complete", "config": settings},
                )
