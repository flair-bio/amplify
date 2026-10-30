"""Output reservation for the generic evaluation runner.

Every ordinary target (main and external test workspaces) is reserved before any
model work, so a rejected overwrite stops the run before fitting. The run
manifest is marked complete only after the body finishes. Each completed run
records its workspace identity (model and dataset) plus the predictor identity
that produced it, because predictors are not part of the shared workspace.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from accelerate.utils import broadcast_object_list

from modules.evaluate.src.artifacts.identity import predictor_identity
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_external_workspace,
)
from modules.evaluate.src.utils.io import log_to_file, update_run_manifest

logger = logging.getLogger(__name__)


def run_identity(
    config: EvaluationConfig, workspace_identity: dict[str, Any]
) -> dict[str, Any]:
    """Workspace identity plus the predictor that produced this run."""
    layerwise = config.steps.layerwise
    return {
        **workspace_identity,
        "predictor": predictor_identity(
            mode=config.steps.mode,
            head=config.steps.tune.head.model_dump(mode="json"),
            layerwise=layerwise.model_dump(mode="json")
            if layerwise.enabled
            else {"enabled": False},
        ),
    }


@contextmanager
def reserve_outputs(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    *,
    overwrite: bool,
) -> Iterator[None]:
    """Reserve all ordinary targets before any model work or replacement."""
    with ExitStack() as stack:
        reserved: list[tuple[EvaluationWorkspace, list[Path]]] = []
        error: Exception | None = None
        if accelerator.is_main_process:
            try:
                workspaces = [workspace]
                if config.steps.mode != "evaluate_as_is":
                    workspaces.extend(
                        get_external_workspace(workspace, external)
                        for external in config.external_test_sets
                    )
                if len({item.run_root.resolve() for item in workspaces}) != len(
                    workspaces
                ):
                    raise ValueError(
                        "Main and external test output labels must differ."
                    )
                for item in workspaces:
                    targets = stack.enter_context(item.reserve_run(overwrite=overwrite))
                    reserved.append((item, targets))
                for item, targets in reserved:
                    item.initialize_run(
                        targets,
                        overwrite=overwrite,
                        config=config.model_dump(mode="json"),
                    )
                    stack.enter_context(log_to_file(item.run_root / "log"))
            except Exception as exception:
                logger.exception("Cannot reserve evaluation outputs")
                error = exception
        messages = [str(error) if error is not None else None]
        if getattr(accelerator, "num_processes", 1) > 1:
            messages = broadcast_object_list(messages, from_process=0)
        if messages[0] is not None:
            if error is not None:
                raise error
            raise RuntimeError(messages[0])
        yield
        if accelerator.is_main_process:
            for item, _ in reserved:
                update_run_manifest(
                    item.run_manifest_path,
                    "evaluation",
                    {
                        "status": "complete",
                        "config": config.model_dump(mode="json"),
                        "identity": run_identity(config, item.pinned_identity()),
                    },
                )
