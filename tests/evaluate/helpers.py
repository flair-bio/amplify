"""Offline fixtures composing the real evaluation services."""

import sys
from dataclasses import dataclass, fields
from types import SimpleNamespace

from accelerate import Accelerator
from torch import nn

from modules.evaluate.src.artifacts.checkpoints import save_predictor
from modules.evaluate.src.artifacts.predictions import persist_predictions
from modules.evaluate.src.artifacts.reports import (
    publish_scores,
    publish_structured_scores,
)
from modules.evaluate.src.artifacts.results import FitResult, StructuredScoreOutput
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.controls.fitting import prepare_control_predictors
from modules.evaluate.src.dataset.types import TestData, TrainingData
from modules.evaluate.src.model.controls import ControlPredictors
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.model.prediction import predict_layerwise, predict_test_set
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.trainers import fitting


@dataclass
class EvaluationInputs:
    predictor: Predictor
    training: TrainingData
    test: TestData
    controls: ControlPredictors
    fit: FitResult | None = None


def make_inputs(*, model, tokenizer, trunk=None, controls=None, **data):
    training_fields = {item.name for item in fields(TrainingData)}
    test_fields = {item.name for item in fields(TestData)}
    unknown = data.keys() - training_fields - test_fields
    if unknown:
        raise TypeError(f"Unknown fixture fields: {sorted(unknown)}")
    return EvaluationInputs(
        Predictor(model, tokenizer, trunk),
        TrainingData(
            **{
                "train_dataloader": None,
                "val_dataloader": None,
                **{key: value for key, value in data.items() if key in training_fields},
            }
        ),
        TestData(
            **{
                "test_dataloader": None,
                **{key: value for key, value in data.items() if key in test_fields},
            }
        ),
        controls if controls is not None else ControlPredictors(),
    )


def fitting_state(config, train_trunk=False, *, mode=None):
    return SimpleNamespace(
        config=config,
        train_trunk=train_trunk or mode == "finetune",
        result=FitResult(model=nn.Identity(), seed=710019),
    )


def patch_function(monkeypatch, original, replacement):
    """Patch the service and its directly imported bindings."""
    for name, module in list(sys.modules.items()):
        if not name.startswith("modules.evaluate.src.") or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, attribute, replacement)


def fit_inputs(
    config,
    workspace,
    inputs,
    *,
    seed=710019,
    accelerator=None,
    mode="tune_probe",
    state=None,
    publish=True,
):
    accelerator = accelerator if accelerator is not None else Accelerator(cpu=True)
    result = fitting.fit_predictor(
        config=config,
        workspace=workspace.config
        if isinstance(workspace, EvaluationWorkspace)
        else workspace,
        predictor=inputs.predictor,
        training=inputs.training,
        accelerator=accelerator,
        seed=seed,
        mode=mode,
    )
    inputs.predictor.model = result.model
    inputs.fit = result
    if state is not None:
        state.result = result
    if publish:
        save_predictor(config, workspace, inputs.predictor, result, accelerator)
    return inputs


def fit_controls(
    config,
    workspace,
    inputs,
    *,
    seed=710019,
    accelerator=None,
    controls=(),
    fitted=True,
):
    accelerator = accelerator if accelerator is not None else Accelerator(cpu=True)
    inputs.controls = prepare_control_predictors(
        config,
        workspace,
        inputs.predictor,
        inputs.training,
        seed=seed,
        accelerator=accelerator,
        controls=controls,
        fitted=fitted,
    )
    return inputs


def predict_inputs(
    config, workspace, inputs, *, seed=710019, accelerator=None, **kwargs
):
    accelerator = accelerator if accelerator is not None else Accelerator(cpu=True)
    output = predict_test_set(
        config,
        workspace,
        inputs.predictor,
        inputs.training,
        inputs.test,
        inputs.controls,
        seed=seed,
        accelerator=accelerator,
        **kwargs,
    )
    if config.persist_to_disk:
        persist_predictions(workspace, output, seed)
    return output


def predict_layer_inputs(config, workspace, inputs, accelerator=None, *, seed=710019):
    accelerator = accelerator if accelerator is not None else Accelerator(cpu=True)
    outputs = predict_layerwise(
        config,
        workspace,
        inputs.predictor,
        inputs.training,
        inputs.test,
        inputs.controls,
        accelerator,
        seed,
    )
    if config.persist_to_disk:
        for layer, output in outputs.items():
            target = get_evaluation_workspace(
                workspace.config,
                variant=f"{workspace.variant}-layer_{layer}",
                seed=workspace.seed,
            )
            if workspace.test_subset is not None:
                target = target.test_workspace(workspace.test_subset)
            persist_predictions(target, output, seed)
    return outputs


def score_inputs(config, workspace, *, predict_result, seed=710019):
    task = get_task(predict_result.metadata.task_type, workspace.num_labels)
    result = task.score(predict_result, config, seed=seed)
    if isinstance(result, StructuredScoreOutput):
        return publish_structured_scores(
            config,
            result=result,
            output_dir=workspace.scores_path,
            manifest_path=workspace.run_manifest_path,
            baseline_references={},
        )
    return publish_scores(config, workspace, result, {}, seed=seed)
