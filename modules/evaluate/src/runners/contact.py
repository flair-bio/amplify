"""Fixed-protocol contact evaluation, independent of flat loader fitting."""

from __future__ import annotations

from functools import partial

from accelerate import Accelerator
from datasets import DatasetDict

from modules.evaluate.src.artifacts.cache import (
    load_tensor_cache,
    validate_tensor_cache,
)
from modules.evaluate.src.artifacts.checkpoints import save_predictor
from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.predictions import persist_predictions
from modules.evaluate.src.artifacts.reports import publish_structured_scores
from modules.evaluate.src.artifacts.results import EvaluationResult, PipelineResult
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace
from modules.evaluate.src.config.evaluation import EvaluationConfig, PrepareConfig
from modules.evaluate.src.controls.contact import (
    build_contact_controls,
    contact_control_settings,
)
from modules.evaluate.src.dataset.baseline_data import contact_baseline_context
from modules.evaluate.src.dataset.contact import (
    PROTOCOL_VERSION,
    ContactDataset,
    load_contact_dataset,
    prepare_contact_dataset,
)
from modules.evaluate.src.dataset.types import FeatureSet
from modules.evaluate.src.model.contact_features import (
    contact_feature_requests,
    contact_fitting_identity,
)
from modules.evaluate.src.model.contact_model import build_contact_model
from modules.evaluate.src.model.contact_prediction import predict_contact
from modules.evaluate.src.model.controls import ControlPredictors
from modules.evaluate.src.model.factory import Predictor, load_tokenizer
from modules.evaluate.src.tasks.contact_prediction import prediction_metadata, score
from modules.evaluate.src.trainers.contact import fit_contact_predictor


def prepare_contact_features(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    dataset: ContactDataset,
) -> tuple[dict[str, FeatureSet], dict]:
    cache = config.embedding_cache
    features = {}
    for name, request in contact_feature_requests(predictor, dataset).items():
        values, fingerprint = load_tensor_cache(
            cache_dir=workspace.embeddings_path
            if cache.cache_policy != "off"
            else None,
            namespace=request.namespace,
            identity=request.identity,
            extract=request.extract,
            validate=partial(
                validate_tensor_cache, shapes=request.shapes, dtype=request.dtype
            ),
            refresh=cache.cache_policy == "refresh",
            max_cache_gb=cache.max_cache_gb,
        )
        features[name] = FeatureSet(values, fingerprint)
    return features, contact_fitting_identity(predictor, dataset, features)


def run_contact_evaluation(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    accelerator: Accelerator,
    *,
    dataset: DatasetDict | None = None,
    subsets: dict[str, DatasetDict] | None = None,
) -> PipelineResult:
    source, model_id = workspace.dataset_repo_id, workspace.model_repo_id
    if source is None or model_id is None:
        raise ValueError(
            "Contact evaluation requires dataset_repo_id and model_repo_id."
        )
    contacts = (
        load_contact_dataset(
            source,
            revision=workspace.config.dataset_revision,
            config_name=workspace.config.dataset_config_name,
        )
        if dataset is None
        else prepare_contact_dataset(
            dataset, source=source, revision=workspace.config.dataset_revision
        )
    )
    tokenizer = load_tokenizer(model_id)
    predictor = Predictor(
        build_contact_model(
            model_id, predictor=config.steps.tune.head.head_type, tokenizer=tokenizer
        ).to(accelerator.device),
        tokenizer,
    )
    workspace.setup_directories()
    if config.steps.mode != "evaluate_as_is":
        features, identity = prepare_contact_features(
            config.steps.prepare, workspace, predictor, contacts
        )
        if identity:
            workspace.validate_experiment("inputs", identity)
        result = fit_contact_predictor(
            predictor,
            contacts,
            features,
            identity,
            config.steps.tune,
            seed=config.seed,
            device=accelerator.device,
        )
        predictor.model = result.model
        save_predictor(config.steps.tune, workspace, predictor, result, accelerator)
    controls = ControlPredictors()
    for name, inputs in build_contact_controls(
        predictor,
        config.steps.predict.controls,
        seed=config.seed,
        device=accelerator.device,
    ).items():
        target = workspace.control_workspace(name)
        target.setup_directories()
        control = Predictor(inputs.model, tokenizer)
        features, identity = prepare_contact_features(
            config.steps.prepare, target, control, contacts
        )
        result = fit_contact_predictor(
            control,
            contacts,
            features,
            identity,
            config.steps.tune,
            seed=inputs.seed,
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            inputs.serialization_model.save_pretrained(target.model_path)
            tokenizer.save_pretrained(target.model_path)
            update_run_manifest(
                workspace.run_manifest_path,
                "fit",
                contact_control_settings(inputs, result),
                namespace=("controls", name),
            )
        controls.independent_models[name] = result.model
    if subsets:
        output = PipelineResult([])
        for index, (name, data) in enumerate(subsets.items()):
            target = workspace.test_workspace(name)
            view = (
                contacts
                if index == 0
                else prepare_contact_dataset(
                    data, source=source, revision=workspace.config.dataset_revision
                )
            )
            evaluation = {
                "status": "running",
                "test_subset": name,
                "filters": config.steps.predict.test_subsets[name],
                "parent_manifest": str(workspace.run_manifest_path),
                "model_path": str(workspace.model_path),
            }
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                evaluation,
                namespace=target.manifest_namespace,
            )
            output.test_subsets[name] = evaluate_contact_test_set(
                config, target, predictor, view, controls
            )
            evaluation["status"] = "complete"
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                evaluation,
                namespace=target.manifest_namespace,
            )
        return output
    return evaluate_contact_test_set(config, workspace, predictor, contacts, controls)


def evaluate_contact_test_set(
    config: EvaluationConfig,
    workspace: EvaluationWorkspace,
    predictor: Predictor,
    dataset: ContactDataset,
    controls: ControlPredictors,
) -> PipelineResult:
    metadata = prediction_metadata(
        {
            "model_name": workspace.model_name,
            "model_repo_id": workspace.model_repo_id,
            "dataset_name": workspace.dataset_name,
            "dataset_repo_id": workspace.dataset_repo_id,
            "dataset_config_name": workspace.config.dataset_config_name,
            "dataset_revision": workspace.config.dataset_revision,
            "dataset_fingerprint": dataset.fingerprint,
            "schema_version": dataset.schema_version,
            "protocol_version": PROTOCOL_VERSION,
            "task_type": workspace.task_type,
            "split": config.steps.prepare.split.name,
            "test_subset": workspace.test_subset,
            "seed": config.seed,
            "head_type": config.steps.tune.head.head_type,
            "variant": workspace.variant,
        }
    )
    prediction = predict_contact(
        predictor, dataset, controls, metadata, config.steps.predict
    )
    prediction.config = config.steps.predict.model_dump(mode="json")
    prediction.baseline_context = contact_baseline_context(
        workspace.config,
        dataset,
        [],
    )
    if config.steps.predict.persist_to_disk:
        persist_predictions(workspace, prediction, config.seed)
    scores = score(prediction, config.steps.score, seed=config.seed)
    published = publish_structured_scores(
        config.steps.score,
        result=scores,
        output_dir=workspace.scores_path,
        manifest_path=workspace.run_manifest_path,
        baseline_references={},
    )
    return PipelineResult([EvaluationResult(workspace, prediction, published)])
