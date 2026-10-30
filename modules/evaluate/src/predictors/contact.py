"""Contact-prediction inference: trained model plus independently fitted random trunk.

CASP contact prediction consumes per-pair scores rather than class
probabilities, and its only supported control is the independently refitted
``random_trunk`` probe. Specialized extraction and scoring stay in
:mod:`modules.evaluate.src.tasks.contact_prediction`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from accelerate import Accelerator

from modules.evaluate.src.artifacts.feature_cache import EmbeddingCacheConfig
from modules.evaluate.src.artifacts.predictions import persist_predictions
from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.evaluation.contracts import (
    FitOutcome,
    FitSummary,
    PredictionBundle,
    RunContext,
)
from modules.evaluate.src.metrics.contact_metrics import CASP_METRICS
from modules.evaluate.src.model.contact_attention import (
    ContactAttentionModel,
    contact_development_attention_fingerprint,
    fit_contact_attention_head,
)
from modules.evaluate.src.model.contact_jacobian import ContactJacobianModel
from modules.evaluate.src.model.contact_model import (
    ContactPredictionModel,
    fit_contact_head,
    fit_random_trunk_contact_control,
)
from modules.evaluate.src.predictors.embeddings import contact_development_features
from modules.evaluate.src.predictors.inference import build_evaluation_metadata
from modules.evaluate.src.schemas.artifacts import (
    ContactScoreOutput,
    EvaluationMetadata,
    PredictOutput,
    PreparedArtifacts,
)
from modules.evaluate.src.tasks.contact_prediction import (
    ContactDataset,
    ContactPredictionTask,
)
from modules.evaluate.src.training.config import TuneConfig
from modules.evaluate.src.utils.io import write_json


def score_contact_predictions(
    predictions: PredictOutput,
    *,
    bootstrap_enabled: bool,
    metrics: list[str] | None,
) -> ContactScoreOutput:
    """Score contact units with the fixed CASP metrics (no flat bootstrap)."""
    if bootstrap_enabled:
        raise ValueError("CASP bootstrap must resample complexes, not flat rows")
    if metrics is not None and set(metrics) != set(CASP_METRICS):
        raise ValueError("Contact scoring reports the fixed CASP metrics")
    return ContactPredictionTask().score_units(predictions)


def predict_contacts(
    workspace: EvaluationWorkspace,
    prepared: PreparedArtifacts,
    config: PredictConfig,
    metadata: EvaluationMetadata,
    seed: int,
    accelerator: Accelerator | None,
) -> PredictOutput:
    """Predict contact maps for the trained model and any requested control."""
    unsupported_controls = set(config.controls) - {"random_trunk"}
    if unsupported_controls:
        raise ValueError(
            "CASP contact prediction only supports the independently "
            f"fitted random_trunk control; got {sorted(unsupported_controls)}"
        )
    if config.collect_probabilities:
        raise ValueError("CASP contacts consume pair scores, not class probabilities")
    if accelerator is not None and accelerator.num_processes != 1:
        raise ValueError("Contact prediction requires a single process")
    if "random_trunk" in config.controls and prepared.random_trunk_model is None:
        raise ValueError(
            "random_trunk was requested but no fitted contact control is available"
        )
    predictor = cast(Any, prepared.model).predict
    task = ContactPredictionTask()
    output = task.predict_units(
        _contact_dataset(prepared),
        lambda chain: predictor(chain, prepared.tokenizer),
        metadata,
    )
    if "random_trunk" in config.controls:
        random_predictor = cast(Any, prepared.random_trunk_model).predict
        control_output = task.predict_units(
            _contact_dataset(prepared),
            lambda chain: random_predictor(chain, prepared.tokenizer),
            metadata,
        )
        output.controls["random_trunk"] = control_output.predictions
        output.requested_controls = ["random_trunk"]
    output.config = config.model_dump(mode="json")
    if config.persist_to_disk:
        persist_predictions(workspace, output, seed)
    return output


def _save_random_trunk(
    random_model: ContactPredictionModel | ContactAttentionModel | ContactJacobianModel,
    control_model_path: Any,
) -> None:
    if isinstance(random_model, ContactJacobianModel):
        random_model.trunk.save_pretrained(control_model_path)
    elif isinstance(random_model, (ContactPredictionModel, ContactAttentionModel)):
        random_model.save_pretrained(control_model_path)
    else:
        raise TypeError(
            f"Unsupported random-trunk model: {type(random_model).__name__}"
        )


def _contact_dataset(prepared: PreparedArtifacts) -> ContactDataset:
    """The CASP contact dataset that contact predictors are fitted and scored on."""
    if prepared.contact_dataset is None:
        raise ValueError("contact prediction requires a prepared contact dataset")
    return prepared.contact_dataset


class ContactPredictor:
    """Shared CASP contact lifecycle: fit one head (if any), an independently
    fitted ``random_trunk`` control on request, then predict the main model and
    that control. Subclasses only choose their head and how it is fitted.
    """

    name: str
    head_type: str
    #: Contact models have no transferable probe; external sets are not scored.
    transfers_to_external = False

    def __init__(
        self,
        *,
        tune: TuneConfig,
        cache: EmbeddingCacheConfig,
        predict: PredictConfig,
        seed: int,
    ) -> None:
        self.tune = tune
        self.cache = cache
        self.predict_config = predict
        self.seed = seed

    def fit_head(
        self, context: RunContext, prepared: PreparedArtifacts, device: Any
    ) -> dict[str, Any] | None:
        """Fit the main head and record it; ``None`` when there is no fitted head."""
        raise NotImplementedError

    def fit(self, context: RunContext, prepared: Any) -> FitOutcome:
        assert isinstance(prepared, PreparedArtifacts)
        assert prepared.contact_dataset is not None
        device = context.accelerator.device
        self.fit_head(context, prepared, device)
        random_model = None
        random_seed = None
        if "random_trunk" in self.predict_config.controls:
            random_model, random_seed = self._fit_random_trunk(
                context.workspace, prepared, device
            )
        fitted = replace(
            prepared, random_trunk_model=random_model, random_trunk_seed=random_seed
        )
        summary = FitSummary(
            predictor=self.name,
            details={
                "head_type": self.head_type,
                "random_trunk": random_model is not None,
            },
        )
        return FitOutcome(summary=summary, fitted=fitted)

    def _fit_random_trunk(
        self, workspace: EvaluationWorkspace, prepared: PreparedArtifacts, device: Any
    ) -> tuple[Any, int]:
        cache_on = self.cache.cache_policy != "off"
        random_model, settings = fit_random_trunk_contact_control(
            cast(ContactPredictionModel, prepared.model),
            _contact_dataset(prepared),
            prepared.tokenizer,
            seed=self.seed,
            max_epochs=self.tune.max_epochs,
            device=device,
            cache_dir=workspace.embeddings_path if cache_on else None,
            refresh_cache=self.cache.cache_policy == "refresh",
            max_cache_gb=self.cache.max_cache_gb,
        )
        control_workspace = workspace.control_workspace("random_trunk")
        control_workspace.setup_directories()
        _save_random_trunk(random_model, control_workspace.model_path)
        prepared.tokenizer.save_pretrained(control_workspace.model_path)
        write_json(control_workspace.stats_path / "contact_training.json", settings)
        return random_model, settings["trunk_seed"]

    def _record_head(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        settings: dict[str, Any],
    ) -> None:
        write_json(workspace.stats_path / "contact_training.json", settings)
        if self.tune.save_best_model:
            cast(Any, prepared.model).save_pretrained(workspace.model_path)
            prepared.tokenizer.save_pretrained(workspace.model_path)
            write_json(workspace.model_path / "tune_config.json", settings)

    def predict(
        self, context: RunContext, prepared: Any
    ) -> tuple[PredictionBundle, ...]:
        assert isinstance(prepared, PreparedArtifacts)
        workspace = context.workspace
        metadata = build_evaluation_metadata(
            workspace,
            split=prepared.split,
            seed=self.seed,
            head_type=self.head_type,
            variant=workspace.variant,
        )
        output = predict_contacts(
            workspace,
            prepared,
            self.predict_config,
            metadata,
            self.seed,
            context.accelerator,
        )
        return (
            PredictionBundle(
                predictor=self.name,
                split=context.split,
                branch="main",
                workspace=workspace,
                payload=output,
            ),
        )


def _validate_contact_inputs(
    workspace: EvaluationWorkspace, dataset: Any, key: str, protocol: str
) -> None:
    workspace.validate_experiment(
        "inputs",
        {
            "dataset_fingerprint": dataset.fingerprint,
            "development_features": key,
            "protocol": protocol,
        },
    )


class ContactEmbeddingPredictor(ContactPredictor):
    """Bilinear pair head over frozen, cached residue embeddings."""

    name = "contact_embedding"
    head_type = "bilinear512"

    def fit_head(
        self, context: RunContext, prepared: PreparedArtifacts, device: Any
    ) -> dict[str, Any]:
        workspace = context.workspace
        dataset = _contact_dataset(prepared)
        model = cast(ContactPredictionModel, prepared.model)
        cache_on = self.cache.cache_policy != "off"
        features, key = contact_development_features(
            model,
            prepared.tokenizer,
            dataset,
            workspace.embeddings_path if cache_on else None,
            refresh=self.cache.cache_policy == "refresh",
            max_cache_gb=self.cache.max_cache_gb,
        )
        _validate_contact_inputs(workspace, dataset, key, "casp-contact-v1")
        settings = fit_contact_head(
            model,
            dataset,
            features,
            seed=self.seed,
            max_epochs=self.tune.max_epochs,
            device=device,
        )
        self._record_head(workspace, prepared, settings)
        return settings


class ContactAttentionPredictor(ContactPredictor):
    """Logistic head over attention-map fingerprints (``attention_l1``)."""

    name = "contact_attention"
    head_type = "attention_l1"

    def fit_head(
        self, context: RunContext, prepared: PreparedArtifacts, device: Any
    ) -> dict[str, Any]:
        del device  # the attention head is fitted on CPU features
        workspace = context.workspace
        dataset = _contact_dataset(prepared)
        model = cast(ContactAttentionModel, prepared.model)
        key = contact_development_attention_fingerprint(
            model, prepared.tokenizer, dataset
        )
        _validate_contact_inputs(workspace, dataset, key, "casp-attention-l1-v1")
        settings = fit_contact_attention_head(
            model,
            dataset,
            prepared.tokenizer,
            seed=self.seed,
        )
        self._record_head(workspace, prepared, settings)
        return settings


class ContactJacobianPredictor(ContactPredictor):
    """Categorical Jacobian of the pretrained model: nothing is fitted."""

    name = "contact_jacobian"
    head_type = "categorical_jacobian"

    def fit_head(
        self, context: RunContext, prepared: PreparedArtifacts, device: Any
    ) -> None:
        del context, prepared, device
