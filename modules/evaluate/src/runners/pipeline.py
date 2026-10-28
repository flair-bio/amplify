"""Prepare -> Tune -> Predict -> Score for one evaluation run, shared by
ordinary evaluation and each cross-validation fold."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from accelerate.utils import broadcast_object_list
from datasets import DatasetDict

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.schemas.artifacts import (
    ContactScoreOutput,
    CrossValidationIdentity,
    PredictOutput,
    PreparedArtifacts,
    ScoreOutput,
)
from modules.evaluate.src.steps.predict import PredictStep
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.steps.tune import TuneStep
from modules.evaluate.src.utils.io import log_to_file, update_run_manifest

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outputs keyed by layer index (``None`` for ordinary single-head runs).
    ``scores`` is populated on the main process only."""

    predictions: dict[int | None, PredictOutput]
    scores: dict[int | None, ScoreOutput | ContactScoreOutput] = field(
        default_factory=dict
    )
    # Results of the same fitted model on each configured external test set.
    external: dict[str, PipelineResult] = field(default_factory=dict)


def run_pipeline(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    dataset: DatasetDict | None = None,
    cross_validation: CrossValidationIdentity | None = None,
    *,
    overwrite: bool = False,
) -> PipelineResult:
    """Run every step once for *workspace*, optionally on a supplied dataset."""
    if config.workspace.task_type == "contact_prediction":
        if (
            cross_validation is not None
            or getattr(accelerator, "num_processes", 1) != 1
        ):
            raise ValueError(
                "Contact embeddings require a single-process fixed-split run"
            )
        if workspace.seed != config.seed:
            raise ValueError(
                "Contact workspace must be created with the configured seed"
            )
    if cross_validation is not None:
        if overwrite:
            raise ValueError(
                "--overwrite is not supported for CV; choose a new run_id."
            )
        return _execute_steps(config, workspace, accelerator, dataset, cross_validation)
    with _reserve_outputs(config, workspace, accelerator, overwrite=overwrite):
        result = _execute_steps(
            config, workspace, accelerator, dataset, cross_validation
        )
        return result


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
                        get_evaluation_workspace(
                            workspace.config.model_copy(
                                update={
                                    "dataset_name": external.name,
                                    "dataset_repo_id": external.dataset_repo_id,
                                }
                            ),
                            variant=workspace.variant,
                            seed=workspace.seed,
                        )
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
    cross_validation: CrossValidationIdentity | None,
) -> PipelineResult:
    if workspace.task_type == "contact_prediction":
        return _execute_contacts(config, workspace, accelerator, dataset)
    logger.info("=== Starting Prepare Step ===")
    prepared_artifacts = PrepareStep(config.steps.prepare).run(
        workspace,
        seed=config.seed,
        accelerator=accelerator,
        layerwise_config=config.steps.layerwise,
        dataset=dataset,
        cross_validation=cross_validation,
        **(
            {"baseline_names": config.baselines.names} if config.baselines.names else {}
        ),
    )
    logger.info("=== Completed Prepare Step ===")

    # Train in tune_probe or finetune mode. evaluate_as_is evaluates the task
    # head loaded by PrepareStep without entering TuneStep.
    mode = config.steps.mode
    tuned = mode != "evaluate_as_is"
    if mode != "evaluate_as_is":
        logger.info("=== Fine-tuning model ===")
        prepared_artifacts = TuneStep(config.steps.tune, mode=mode).run(
            workspace,
            prepared_artifacts,
            seed=config.seed,
            accelerator=accelerator,
            controls=config.steps.predict.controls,
        )
        logger.info("=== Completed Fine-tune Step ===")
    else:
        logger.info(
            "=== Skipping Tune Step (steps.mode=evaluate_as_is; "
            "hyperparameter search not applicable) ==="
        )
        # evaluate_as_is returns a raw model from PrepareStep, so it still
        # needs accelerator.prepare() before Predict/Score can use it.
        prepared_artifacts.model = accelerator.prepare(prepared_artifacts.model)

    result = _predict_and_score(
        config, workspace, prepared_artifacts, accelerator, tuned=tuned
    )

    if config.external_test_sets and not tuned:
        logger.warning(
            "Skipping external_test_sets: steps.mode=evaluate_as_is has no "
            "fitted probe to transfer."
        )
    elif config.external_test_sets:
        prepare_step = PrepareStep(config.steps.prepare)
        for external in config.external_test_sets:
            logger.info("=== Starting external test set '%s' ===", external.name)
            external_prepared = prepare_step.prepare_external_test(
                workspace,
                prepared_artifacts,
                external,
                accelerator=accelerator,
                seed=config.seed,
                layerwise_config=config.steps.layerwise,
            )
            external_workspace = get_evaluation_workspace(
                workspace.config.model_copy(
                    update={
                        "dataset_name": external.name,
                        "dataset_repo_id": external.dataset_repo_id,
                    }
                ),
                variant=workspace.variant,
                seed=workspace.seed,
            )
            external_workspace.setup_directories()
            result.external[external.name] = _predict_and_score(
                config, external_workspace, external_prepared, accelerator, tuned=tuned
            )
            logger.info("=== Completed external test set '%s' ===", external.name)
    accelerator.wait_for_everyone()
    return result


def _execute_contacts(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Any,
    dataset: DatasetDict | None,
) -> PipelineResult:
    from transformers import AutoTokenizer

    from modules.evaluate.src.model.contact_model import build_contact_model
    from modules.evaluate.src.tasks.contact_prediction import prepare_contact_dataset

    source = workspace.config.dataset_repo_id
    if source is None or workspace.config.model_repo_id is None:
        raise ValueError(
            "Contact prediction requires dataset_repo_id and model_repo_id"
        )
    if dataset is None:
        contacts = PrepareStep(config.steps.prepare).run_contacts(
            source,
            revision=workspace.config.dataset_revision,
            edition=workspace.config.edition,
        )
    else:
        contacts = prepare_contact_dataset(
            dataset,
            source=source,
            revision=workspace.config.dataset_revision,
            edition=workspace.config.edition,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        workspace.config.model_repo_id, trust_remote_code=True
    )
    head_type = config.steps.tune.head.head_type
    model = build_contact_model(
        workspace.config.model_repo_id,
        predictor=head_type,
        tokenizer=tokenizer,
    ).to(accelerator.device)
    if head_type != "categorical_jacobian":
        TuneStep(config.steps.tune).run_contacts(
            workspace,
            contacts,
            model,
            tokenizer,
            seed=config.seed,
            device=accelerator.device,
            cache_config=config.steps.prepare.embedding_cache,
        )
    prepared = PreparedArtifacts(
        model=model,
        tokenizer=tokenizer,
        contact_dataset=contacts,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
    )
    predictions = PredictStep(config.steps.predict).run(
        workspace,
        prepared,
        seed=config.seed,
        head_type=head_type,
        variant=workspace.variant,
        accelerator=accelerator,
    )
    scores = ScoreStep(config.steps.score).run_contacts(
        predictions, output_dir=workspace.scores_path
    )
    return PipelineResult(predictions={None: predictions}, scores={None: scores})


def _predict_and_score(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    prepared_artifacts: PreparedArtifacts,
    accelerator: Any,
    *,
    tuned: bool,
) -> PipelineResult:
    """Predict then score on *prepared_artifacts*' test loader."""
    logger.info("=== Starting Predict Step ===")
    predict_step = PredictStep(config.steps.predict)
    layer_predictions: dict[int, PredictOutput] = {}
    predictions: dict[int | None, PredictOutput]
    if config.steps.layerwise.enabled:
        layer_predictions = predict_step.run_layerwise(
            workspace, prepared_artifacts, accelerator, config.seed
        )
        predictions = dict(layer_predictions.items())
    else:
        predictions = {
            None: predict_step.run(
                workspace,
                prepared_artifacts,
                seed=config.seed,
                accelerator=accelerator,
                # head_type doesn't apply to the "pretrained" (steps.tune never
                # ran) condition -- there's no trained head to name.
                head_type=config.steps.tune.head.head_type if tuned else None,
                variant=workspace.variant,
            )
        }
    logger.info("=== Completed Predict Step ===")

    result = PipelineResult(predictions)
    # ScoreStep is a plain, single-process step, so only rank 0 runs it.
    # Passing PredictOutput avoids a disk round trip; ScoreStep still supports
    # manifest-based reloading for standalone or restarted scoring.
    if accelerator.is_main_process:
        logger.info("=== Starting Score Step ===")
        score_step = ScoreStep(config.steps.score)
        for layer, prediction in predictions.items():
            # Separate layer workspaces keep per-depth scores and manifests apart.
            score_workspace = (
                workspace
                if layer is None
                else get_evaluation_workspace(
                    workspace.config,
                    variant=f"{workspace.variant}-layer_{layer}",
                    seed=workspace.seed,
                )
            )
            score = score_step.run(
                score_workspace, seed=config.seed, predict_result=prediction
            )
            result.scores[layer] = score
            label = "Scores" if layer is None else f"Layer {layer} scores"
            logger.info("%s: %s", label, score.scores)
            if score.summary_path is not None:
                logger.info(
                    "Bootstrap confidence intervals written to %s", score.summary_path
                )
        if layer_predictions and config.steps.score.bootstrap_enabled:
            comparison_path = score_step.compare_layers(
                workspace, layer_predictions, seed=config.seed
            )
            if comparison_path is not None:
                logger.info(
                    "Paired cross-layer comparison written to %s", comparison_path
                )
        logger.info("=== Completed Score Step ===")
    return result
