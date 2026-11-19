"""Fit flat predictors once and reuse them across independent test views."""

from __future__ import annotations

import logging

from accelerate import Accelerator
from datasets import DatasetDict

from modules.evaluate.src.artifacts.checkpoints import save_predictor
from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.predictions import persist_predictions
from modules.evaluate.src.artifacts.reports import (
    publish_layer_comparison,
    publish_scores,
)
from modules.evaluate.src.artifacts.results import (
    EvaluationResult,
    PipelineResult,
    PredictOutput,
)
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
)
from modules.evaluate.src.config import (
    EvaluationConfig,
    ExternalTestSetConfig,
)
from modules.evaluate.src.controls.fitting import prepare_control_predictors
from modules.evaluate.src.dataset.preparation import (
    prepare_test_data,
    prepare_training_data,
)
from modules.evaluate.src.dataset.types import TestData, TrainingData
from modules.evaluate.src.model.controls import ControlPredictors
from modules.evaluate.src.model.factory import (
    Predictor,
    build_predictor,
    build_probe,
    load_tokenizer,
)
from modules.evaluate.src.model.prediction import predict_layerwise, predict_test_set
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.trainers.fitting import fit_predictor

logger = logging.getLogger(__name__)


def run_flat_evaluation(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Accelerator,
    *,
    dataset: DatasetDict | None = None,
    subsets: dict[str, DatasetDict] | None = None,
) -> PipelineResult:
    task = get_flat_task(workspace.task_type, workspace.num_labels)
    model_id = workspace.model_repo_id or workspace.model_name
    predictor = Predictor(
        build_predictor(
            model_id,
            workspace.task_type,
            task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
            workspace.model_kwargs,
        ).to(accelerator.device),
        load_tokenizer(model_id),
    )
    training, test = prepare_training_data(
        config.steps.prepare,
        workspace,
        predictor.model,
        predictor.tokenizer,
        seed=config.seed,
        accelerator=accelerator,
        layerwise_config=config.steps.layerwise,
        dataset=dataset,
        group_column=config.steps.score.group_column,
        validation_group_column=config.steps.tune.hyperparameter_search.group_column,
    )
    if training.embedding is not None:
        predictor.trunk = predictor.model
    predictor.model = build_probe(
        predictor.model,
        predictor.tokenizer,
        workspace.task_type,
        task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
        embedding_size=training.embedding.backend.hidden_size
        if training.embedding is not None
        else None,
        layerwise_config=config.steps.layerwise,
    ).to(accelerator.device)
    tuned = config.steps.mode != "evaluate_as_is"
    if config.steps.mode != "evaluate_as_is":
        fit = fit_predictor(
            workspace=workspace.config,
            predictor=predictor,
            training=training,
            config=config.steps.tune,
            accelerator=accelerator,
            seed=config.seed,
            mode=config.steps.mode,
        )
        predictor.model = fit.model
        save_predictor(config.steps.tune, workspace, predictor, fit, accelerator)
    else:
        predictor.model = accelerator.prepare(predictor.model)
    controls = prepare_control_predictors(
        config.steps.tune,
        workspace,
        predictor,
        training,
        seed=config.seed,
        accelerator=accelerator,
        controls=config.steps.predict.controls,
        fitted=tuned,
    )
    if subsets:
        result = PipelineResult([])
        for index, (name, subset_data) in enumerate(subsets.items()):
            target = workspace.test_workspace(name)
            view = test
            if index > 0:
                subset_source = ExternalTestSetConfig(
                    name=name,
                    dataset_repo_id=workspace.dataset_repo_id or workspace.dataset_name,
                    dataset_config_name=workspace.config.dataset_config_name,
                    dataset_revision=workspace.config.dataset_revision,
                    split=config.steps.prepare.split.test,
                )
                view = prepare_test_data(
                    config.steps.prepare,
                    workspace,
                    predictor.trunk,
                    predictor.tokenizer,
                    test,
                    subset_source,
                    accelerator=accelerator,
                    seed=config.seed,
                    layerwise_config=config.steps.layerwise,
                    dataset=DatasetDict({"test": subset_data["test"]}),
                    group_column=config.steps.score.group_column,
                )
            evaluation = {
                "status": "running",
                "test_subset": name,
                "filters": config.steps.predict.test_subsets[name],
                "parent_manifest": str(workspace.run_manifest_path),
                "model_path": str(workspace.model_path),
            }
            if accelerator.is_main_process:
                update_run_manifest(
                    workspace.run_manifest_path,
                    "evaluation",
                    evaluation,
                    namespace=target.manifest_namespace,
                )
            result.test_subsets[name] = evaluate_test_set(
                config,
                target,
                predictor,
                training,
                view,
                controls,
                accelerator,
                tuned=tuned,
            )
            if accelerator.is_main_process:
                evaluation["status"] = "complete"
                update_run_manifest(
                    workspace.run_manifest_path,
                    "evaluation",
                    evaluation,
                    namespace=target.manifest_namespace,
                )
    else:
        result = evaluate_test_set(
            config,
            workspace,
            predictor,
            training,
            test,
            controls,
            accelerator,
            tuned=tuned,
        )
    if config.external_test_sets and not tuned:
        logger.warning("Skipping external_test_sets for steps.mode=evaluate_as_is.")
    if tuned:
        for external in config.external_test_sets:
            view = prepare_test_data(
                config.steps.prepare,
                workspace,
                predictor.trunk,
                predictor.tokenizer,
                test,
                external,
                accelerator=accelerator,
                seed=config.seed,
                layerwise_config=config.steps.layerwise,
                group_column=config.steps.score.group_column,
            )
            result.external[external.name] = evaluate_test_set(
                config,
                workspace.external_workspace(external),
                predictor,
                training,
                view,
                controls,
                accelerator,
                tuned=True,
            )
    accelerator.wait_for_everyone()
    return result


def evaluate_test_set(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    training: TrainingData,
    test: TestData,
    controls: ControlPredictors,
    accelerator: Accelerator,
    *,
    tuned: bool,
) -> PipelineResult:
    layer_predictions: dict[int, PredictOutput] = {}
    if config.steps.layerwise.enabled:
        layer_predictions = predict_layerwise(
            config.steps.predict,
            workspace,
            predictor,
            training,
            test,
            controls,
            accelerator,
            config.seed,
        )
        evaluations = []
        for layer, prediction in layer_predictions.items():
            target = EvaluationWorkspace(
                workspace.config,
                variant=f"{workspace.variant}-layer_{layer}",
                seed=workspace.seed,
            )
            if workspace.test_subset is not None:
                target = target.test_workspace(workspace.test_subset)
            evaluations.append(EvaluationResult(target, prediction))
    else:
        prediction = predict_test_set(
            config.steps.predict,
            workspace,
            predictor,
            training,
            test,
            controls,
            seed=config.seed,
            head_type=config.steps.tune.head.head_type if tuned else None,
            variant=workspace.variant,
            accelerator=accelerator,
        )
        evaluations = [EvaluationResult(workspace, prediction)]
    if accelerator.is_main_process:
        task = get_flat_task(workspace.task_type, workspace.num_labels)
        for evaluation in evaluations:
            if config.steps.predict.persist_to_disk:
                persist_predictions(
                    evaluation.workspace, evaluation.prediction, config.seed
                )
            scores = task.score(
                evaluation.prediction, config.steps.score, seed=config.seed
            )
            evaluation.score = publish_scores(
                config.steps.score, evaluation.workspace, scores, {}, seed=config.seed
            )
        if layer_predictions and config.steps.score.bootstrap_enabled:
            publish_layer_comparison(
                config.steps.score, workspace, layer_predictions, seed=config.seed
            )
    return PipelineResult(evaluations)
