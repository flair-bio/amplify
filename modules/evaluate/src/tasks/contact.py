"""CASP contact prediction on the generic lifecycle.

The task prepares HF-only contact chains, loads the pretrained model with the
requested contact head, and scores with the fixed CASP metrics. Which head is
fitted, and how, belongs to the predictor passed in (see
:mod:`modules.evaluate.src.predictors.contact`). Specialized extraction, unit
scoring and publication stay in the contact owners this module calls.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

from modules.evaluate.src.artifacts.scores import publish_contact_scores
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.evaluation.contracts import (
    PredictionBundle,
    Predictor,
    RunContext,
    ScoringResult,
)
from modules.evaluate.src.model.contact_model import build_contact_model
from modules.evaluate.src.predictors.contact import (
    ContactPredictor,
    score_contact_predictions,
)
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.tasks.contact_prediction import (
    ContactPredictionTask,
    prepare_contact_dataset,
)


class ContactTask:
    """One contact job: single-process, fixed split, one contact head."""

    name = "contact_prediction"
    requires_single_process = True

    def __init__(
        self,
        config: EvaluationConfig,
        predictor_type: type[ContactPredictor],
    ) -> None:
        self.config = config
        self.predictor_type = predictor_type

    def prepare(self, context: RunContext) -> PreparedArtifacts:
        workspace = context.workspace
        if workspace.seed != self.config.seed:
            raise ValueError(
                "Contact workspace must be created with the configured seed"
            )
        source = workspace.config.dataset_repo_id
        model_repo = workspace.config.model_repo_id
        if source is None or model_repo is None:
            raise ValueError(
                "Contact prediction requires dataset_repo_id and model_repo_id"
            )
        if workspace.config.dataset_subset is not None:
            raise ValueError("Contact prediction does not support dataset subsets")
        revision = workspace.config.dataset_revision
        model_revision = workspace.config.model_revision
        edition = workspace.config.edition
        if context.dataset is None:
            dataset = ContactPredictionTask().prepare_dataset(
                source, revision=revision, edition=edition
            )
        else:
            dataset = prepare_contact_dataset(
                context.dataset, source=source, revision=revision, edition=edition
            )
        tokenizer_kwargs = (
            {} if model_revision is None else {"revision": model_revision}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_repo, trust_remote_code=True, **tokenizer_kwargs
        )
        model = build_contact_model(
            model_repo,
            predictor=self.predictor_type.head_type,
            tokenizer=tokenizer,
            revision=model_revision,
        ).to(context.accelerator.device)
        return PreparedArtifacts(
            model=model,
            tokenizer=tokenizer,
            contact_dataset=dataset,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=None,
        )

    def prepare_external(self, context: RunContext, fitted: Any, external: Any) -> Any:
        # Never reached: contact predictors transfer nothing to external views.
        raise RuntimeError("contact_prediction has no external views to prepare.")

    def predictor(self, context: RunContext) -> Predictor:
        return self.predictor_type(
            tune=self.config.steps.tune,
            cache=self.config.steps.prepare.embedding_cache,
            predict=self.config.steps.predict,
            seed=self.config.seed,
        )

    def score(self, context: RunContext, bundle: PredictionBundle) -> ScoringResult:
        score_config = self.config.steps.score
        result = score_contact_predictions(
            bundle.payload,
            bootstrap_enabled=score_config.bootstrap_enabled,
            metrics=score_config.metrics,
        )
        publish_contact_scores(
            result, bundle.workspace.scores_path, score_config.model_dump()
        )
        return ScoringResult(
            predictor=bundle.predictor,
            split=bundle.split,
            branch=bundle.branch,
            payload=result,
        )

    def finalize_view(
        self,
        context: RunContext,
        predictions: tuple[PredictionBundle, ...],
        scores: tuple[ScoringResult, ...],
    ) -> None:
        del context, predictions, scores
