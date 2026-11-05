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
    TestSubsets,
)
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
from modules.evaluate.src.utils.io import update_run_manifest

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
                if _supports_external_tests(config, workspace):
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
    prepare_step = PrepareStep(config.steps.prepare)
    subsets: dict[str, DatasetDict] = {}
    if config.steps.predict.test_subsets:
        subsets = prepare_step.resolve_test_subsets(
            workspace,
            config.steps.predict.test_subsets,
            seed=config.seed,
            dataset=dataset,
        )
        dataset = next(iter(subsets.values()))
    prepared = prepare_step.run(
        workspace,
        seed=config.seed,
        accelerator=accelerator,
        layerwise_config=config.steps.layerwise,
        dataset=dataset,
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
        result = _evaluate_test_subsets(
            config,
            workspace,
            prepared,
            accelerator,
            subsets,
            config.steps.predict.test_subsets,
            tuned=tuned,
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


def _supports_external_tests(
    config: EvaluationConfig, workspace: EvaluationWorkspace
) -> bool:
    return (
        config.steps.mode != "evaluate_as_is"
        or get_task(workspace.task_type).supports_zero_shot_transfer
    )


def _evaluate_test_subsets(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    accelerator: Any,
    subsets: dict[str, DatasetDict],
    filters: TestSubsets,
    *,
    tuned: bool,
    external: ExternalTestSetConfig | None = None,
) -> PipelineResult:
    """Publish named selections identically for main and external test sources."""
    result = PipelineResult([])
    prepare_step = PrepareStep(config.steps.prepare)
    for index, (name, subset_data) in enumerate(subsets.items()):
        subset_workspace = workspace.test_workspace(name)
        subset_workspace.setup_directories()
        evaluation = {
            "status": "running",
            "test_subset": name,
            "filters": filters[name],
            "parent_manifest": str(workspace.run_manifest_path),
            "model_path": str(workspace.model_path),
        }
        if accelerator.is_main_process:
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                evaluation,
                namespace=subset_workspace.manifest_namespace,
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
                external=external,
            )
        )
        result.test_subsets[name] = _predict_and_score(
            config, subset_workspace, subset_prepared, accelerator, tuned=tuned
        )
        if accelerator.is_main_process:
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                {**evaluation, "status": "complete"},
                namespace=subset_workspace.manifest_namespace,
            )
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
    supported = _supports_external_tests(config, workspace)
    if config.external_test_sets and not supported:
        logger.warning(
            "Skipping external_test_sets: steps.mode=evaluate_as_is has no "
            "fitted probe to transfer."
        )
    if not supported or not config.external_test_sets:
        return {}
    results = {}
    prepare_step = PrepareStep(config.steps.prepare)
    for external in config.external_test_sets:
        logger.info("Evaluating external test set '%s'", external.name)
        subsets = (
            prepare_step.resolve_test_subsets(
                workspace,
                external.test_subsets,
                seed=config.seed,
                external=external,
            )
            if external.test_subsets
            else {}
        )
        external_prepared = prepare_step.prepare_external_test(
            workspace,
            prepared,
            external,
            accelerator=accelerator,
            seed=config.seed,
            layerwise_config=config.steps.layerwise,
            dataset=next(iter(subsets.values()), None),
        )
        external_workspace = _external_workspace(workspace, external)
        external_workspace.model_path = workspace.model_path
        external_workspace.setup_directories()
        results[external.name] = (
            _evaluate_test_subsets(
                config,
                external_workspace,
                external_prepared,
                accelerator,
                subsets,
                external.test_subsets,
                tuned=tuned,
                external=external,
            )
            if subsets
            else _predict_and_score(
                config, external_workspace, external_prepared, accelerator, tuned=tuned
            )
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
    # Score the predictions produced by this run.
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
