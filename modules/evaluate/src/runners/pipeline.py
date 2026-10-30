"""Prepare -> optional Tune -> Controls -> Predict -> Score."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from accelerate.utils import broadcast_object_list
from datasets import DatasetDict

from modules.evaluate.src.contracts.artifacts import (
    EvaluationResult,
    PipelineResult,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.contracts.config import (
    EvaluationConfig,
    ExternalTestSetConfig,
)
from modules.evaluate.src.dataset.contact import prepare_contact_dataset
from modules.evaluate.src.dataset.dataloader import resolve_evaluation_data
from modules.evaluate.src.dataset.subsets import select_test_subsets
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.steps.controls import ControlsStep
from modules.evaluate.src.steps.predict import PredictStep
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.steps.tune import TuneStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.io import log_to_file, update_run_manifest

logger = logging.getLogger(__name__)


def run_pipeline(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    dataset: DatasetDict | None = None,
    *,
    overwrite: bool = False,
) -> PipelineResult:
    """Run every step once for *workspace*, optionally on a supplied dataset."""
    get_task(config.workspace.task_type).validate_pipeline(
        config,
        workspace_seed=workspace.seed,
        num_processes=getattr(accelerator, "num_processes", 1),
    )
    with _reserve_outputs(config, workspace, accelerator, overwrite=overwrite):
        return _execute_steps(config, workspace, accelerator, dataset)


def _external_workspace(
    workspace: EvaluationWorkspace, external: ExternalTestSetConfig
) -> EvaluationWorkspace:
    return get_evaluation_workspace(
        workspace.config.model_copy(
            update={
                "dataset_name": external.name,
                "dataset_repo_id": external.dataset_repo_id,
                "dataset_config_name": external.dataset_config_name,
                "dataset_revision": external.dataset_revision,
            }
        ),
        variant=workspace.variant,
        seed=workspace.seed,
    )


@contextmanager
def _reserve_outputs(
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
                        _external_workspace(workspace, external)
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
                    {"status": "complete", "config": config.model_dump(mode="json")},
                )


def _execute_steps(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    dataset: DatasetDict | None,
) -> PipelineResult:
    logger.info("Preparing evaluation inputs")
    subsets: dict[str, DatasetDict] = {}
    if config.steps.predict.test_subsets:
        dataset = resolve_evaluation_data(
            workspace.dataset_repo_id or workspace.dataset_name,
            split=config.steps.prepare.split,
            seed=config.seed,
            validation_fraction=config.steps.prepare.validation_fraction,
            dataset=dataset,
            config_name=config.workspace.dataset_config_name,
            revision=config.workspace.dataset_revision,
        )
        if workspace.task_type == "contact_prediction":
            prepare_contact_dataset(dataset)
        subsets = select_test_subsets(
            dataset,
            config.steps.predict.test_subsets,
            group_column="record_id"
            if workspace.task_type == "contact_prediction"
            else None,
        )
        dataset = next(iter(subsets.values()))
    prepared = PrepareStep(config.steps.prepare).run(
        workspace,
        seed=config.seed,
        accelerator=accelerator,
        layerwise_config=config.steps.layerwise,
        dataset=dataset,
        **(
            {"baseline_names": config.baselines.names} if config.baselines.names else {}
        ),
        head_type=config.steps.tune.head.head_type,
    )

    mode = config.steps.mode
    tuned = mode != "evaluate_as_is"
    if mode != "evaluate_as_is":
        logger.info("Fitting predictor (%s)", mode)
        prepared = TuneStep(config.steps.tune, mode=mode).run(
            workspace,
            prepared,
            seed=config.seed,
            accelerator=accelerator,
        )
    else:
        logger.info(
            "=== Skipping Tune Step (steps.mode=evaluate_as_is; "
            "hyperparameter search not applicable) ==="
        )
        # evaluate_as_is returns a raw model from PrepareStep, so it still
        # needs accelerator.prepare() before Predict/Score can use it.
        prepared.model = accelerator.prepare(prepared.model)

    logger.info("Preparing independent controls")
    prepared = ControlsStep(config.steps.tune, config.steps.prepare).run(
        workspace,
        prepared,
        seed=config.seed,
        accelerator=accelerator,
        controls=config.steps.predict.controls,
        fitted=tuned,
    )
    if subsets:
        result = PipelineResult([])
        prepare_step = PrepareStep(config.steps.prepare)
        for index, (name, subset_data) in enumerate(subsets.items()):
            subset_workspace = workspace.test_workspace(name)
            subset_workspace.setup_directories()
            evaluation = {
                "status": "running",
                "test_subset": name,
                "filters": config.steps.predict.test_subsets[name],
                "parent_manifest": str(workspace.run_manifest_path),
                "model_path": str(workspace.model_path),
            }
            if accelerator.is_main_process:
                update_run_manifest(
                    subset_workspace.run_manifest_path, "evaluation", evaluation
                )
            subset_prepared = (
                prepared
                if index == 0
                else prepare_step.prepare_test_subset(
                    workspace,
                    prepared,
                    name,
                    subset_data,
                    accelerator=accelerator,
                    seed=config.seed,
                    layerwise_config=config.steps.layerwise,
                )
            )
            result.test_subsets[name] = _predict_and_score(
                config, subset_workspace, subset_prepared, accelerator, tuned=tuned
            )
            if accelerator.is_main_process:
                update_run_manifest(
                    subset_workspace.run_manifest_path,
                    "evaluation",
                    {**evaluation, "status": "complete"},
                )
        if accelerator.is_main_process:
            update_run_manifest(
                workspace.run_manifest_path,
                "tests",
                {
                    name: str(workspace.test_workspace(name).run_manifest_path)
                    for name in subsets
                },
            )
    else:
        result = _predict_and_score(
            config, workspace, prepared, accelerator, tuned=tuned
        )
    result.external = _evaluate_external_tests(
        config, workspace, prepared, accelerator, tuned=tuned
    )
    accelerator.wait_for_everyone()
    return result


def _evaluate_external_tests(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Any,
    *,
    tuned: bool,
) -> dict[str, PipelineResult]:
    """Evaluate the same fitted predictor without repeating tuning or controls."""
    if config.external_test_sets and not tuned:
        logger.warning(
            "Skipping external_test_sets: steps.mode=evaluate_as_is has no "
            "fitted probe to transfer."
        )
    if not tuned or not config.external_test_sets:
        return {}
    results = {}
    prepare_step = PrepareStep(config.steps.prepare)
    for external in config.external_test_sets:
        logger.info("Evaluating external test set '%s'", external.name)
        external_prepared = prepare_step.prepare_external_test(
            workspace,
            prepared,
            external,
            accelerator=accelerator,
            seed=config.seed,
            layerwise_config=config.steps.layerwise,
        )
        external_workspace = _external_workspace(workspace, external)
        external_workspace.setup_directories()
        results[external.name] = _predict_and_score(
            config, external_workspace, external_prepared, accelerator, tuned=tuned
        )
    return results


def _predict_and_score(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Any,
    *,
    tuned: bool,
) -> PipelineResult:
    """Predict on every rank, then score gathered observations on rank zero."""
    logger.info("Predicting test observations")
    predict_step = PredictStep(config.steps.predict)
    layer_predictions: dict[int, PredictOutput] = {}
    if config.steps.layerwise.enabled:
        layer_predictions = predict_step.run_layerwise(
            workspace, prepared, accelerator, config.seed
        )
        evaluations = []
        for layer, prediction in layer_predictions.items():
            layer_workspace = get_evaluation_workspace(
                workspace.config,
                variant=f"{workspace.variant}-layer_{layer}",
                seed=workspace.seed,
            )
            if workspace.test_subset is not None:
                layer_workspace = layer_workspace.test_workspace(workspace.test_subset)
            evaluations.append(EvaluationResult(layer_workspace, prediction))
    else:
        prediction = predict_step.run(
            workspace,
            prepared,
            seed=config.seed,
            accelerator=accelerator,
            # head_type doesn't apply to the "pretrained" (steps.tune never
            # ran) condition -- there's no trained head to name.
            head_type=(
                config.steps.tune.head.head_type
                if tuned or get_task(workspace.task_type).uses_predictor_identity
                else None
            ),
            variant=workspace.variant,
        )
        evaluations = [EvaluationResult(workspace, prediction)]

    result = PipelineResult(evaluations)
    # ScoreStep is a plain, single-process step, so only rank 0 runs it.
    # Passing PredictOutput avoids a disk round trip; ScoreStep still supports
    # manifest-based reloading for standalone or restarted scoring.
    if accelerator.is_main_process:
        logger.info("Scoring test observations")
        score_step = ScoreStep(config.steps.score)
        for evaluation in result.evaluations:
            evaluation.score = score_step.run(
                evaluation.workspace,
                seed=config.seed,
                predict_result=evaluation.prediction,
            )
        if layer_predictions and config.steps.score.bootstrap_enabled:
            comparison_path = score_step.compare_layers(
                workspace, layer_predictions, seed=config.seed
            )
            if comparison_path is not None:
                logger.info(
                    "Paired cross-layer comparison written to %s", comparison_path
                )
    return result
