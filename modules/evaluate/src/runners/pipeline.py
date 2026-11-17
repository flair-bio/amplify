"""Coordinate the ordinary evaluation lifecycle and run-wide output ownership."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import DatasetDict

from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.results import PipelineResult
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config.evaluation import EvaluationConfig
from modules.evaluate.src.dataset.loading import resolve_evaluation_data
from modules.evaluate.src.dataset.subsets import select_test_subsets
from modules.evaluate.src.runners.contact import run_contact_evaluation
from modules.evaluate.src.runners.flat import run_flat_evaluation
from modules.evaluate.src.runners.runtime import Runtime
from modules.evaluate.src.tasks.contact_prediction import validate_pipeline

logger = logging.getLogger(__name__)


def run_evaluation(
    config: EvaluationConfig,
    runtime: Runtime,
    dataset: DatasetDict | None = None,
    *,
    overwrite: bool = False,
) -> PipelineResult:
    accelerator = runtime.accelerator
    if accelerator is None:
        raise ValueError("Ordinary evaluation requires an accelerator.")

    workspace = EvaluationWorkspace(
        config.workspace, variant=config.variant, seed=config.seed
    )
    is_contact = workspace.task_type == "contact_prediction"

    if is_contact:
        validate_pipeline(
            config,
            workspace_seed=workspace.seed,
            num_processes=accelerator.num_processes,
        )
        runner = run_contact_evaluation
    else:
        runner = run_flat_evaluation

    with reserve_outputs(config, workspace, accelerator, overwrite=overwrite):
        dataset, subsets = _prepare_test_subsets(config, workspace, dataset)
        return runner(config, workspace, accelerator, dataset=dataset, subsets=subsets)


def _prepare_test_subsets(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    dataset: DatasetDict | None,
) -> tuple[DatasetDict | None, dict[str, DatasetDict]]:
    requested_subsets = config.steps.predict.test_subsets
    if not requested_subsets:
        return dataset, {}

    prepare = config.steps.prepare
    dataset = resolve_evaluation_data(
        workspace.dataset_repo_id or workspace.dataset_name,
        split=prepare.split,
        seed=config.seed,
        validation_fraction=prepare.validation_fraction,
        dataset=dataset,
        config_name=workspace.config.dataset_config_name,
        revision=workspace.config.dataset_revision,
    )

    is_contact = workspace.task_type == "contact_prediction"
    if is_contact:
        from modules.evaluate.src.dataset.contact import prepare_contact_dataset

        prepare_contact_dataset(dataset)

    subsets = select_test_subsets(
        dataset,
        requested_subsets,
        group_column="record_id" if is_contact else None,
    )
    first_subset = next(iter(subsets.values()))
    return first_subset, subsets


@contextmanager
def reserve_outputs(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Accelerator,
    *,
    overwrite: bool,
) -> Iterator[None]:
    """Reserve every destination before fitting, propagating rank-zero failures."""
    with ExitStack() as stack:
        reserved: list[tuple[EvaluationWorkspace, list[Path]]] = []
        error: Exception | None = None
        if accelerator.is_main_process:
            try:
                workspaces = [workspace]
                if config.steps.mode != "evaluate_as_is":
                    workspaces.extend(
                        workspace.external_workspace(external)
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
            except Exception as exception:
                error = exception
        messages = [str(error) if error is not None else None]
        if accelerator.num_processes > 1:
            messages = broadcast_object_list(messages, from_process=0)
        if messages[0] is not None:
            if error is not None:
                raise error
            raise RuntimeError(messages[0])
        try:
            yield
        except BaseException as exception:
            if accelerator.is_main_process:
                for item, _ in reserved:
                    try:
                        update_run_manifest(
                            item.run_manifest_path,
                            "evaluation",
                            {"status": "failed", "error": str(exception)},
                        )
                    except Exception:
                        logger.exception("Could not record evaluation failure")
            raise
        else:
            if accelerator.is_main_process:
                for item, _ in reserved:
                    update_run_manifest(
                        item.run_manifest_path,
                        "evaluation",
                        {
                            "status": "complete",
                            "config": config.model_dump(mode="json"),
                        },
                    )
