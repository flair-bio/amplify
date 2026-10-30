"""Frozen-trunk sequence regression and classification as a generic-runner task.

The task owns the semantics of these families: preparing their observations,
choosing the cached probe predictor, and scoring with the configured metrics.
Model-independent preparation, fitting, inference, and publication are delegated
to their owners in predictors/, training/, scoring/ and artifacts/.
"""

from __future__ import annotations

import logging
from typing import Any

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.evaluation.contracts import (
    PredictionBundle,
    Predictor,
    RunContext,
    ScoringResult,
)
from modules.evaluate.src.predictors.cached_probe import (
    CachedProbePredictor,
    FinetunedProbePredictor,
    LayerwiseCachedProbePredictor,
)
from modules.evaluate.src.predictors.pretrained import PretrainedPredictor
from modules.evaluate.src.predictors.probe_preparation import (
    prepare_external_probe_artifacts,
    prepare_probe_artifacts,
)
from modules.evaluate.src.scoring.run import compare_layers, score_bundle

logger = logging.getLogger(__name__)


class FrozenProbeTask:
    """Sequence and token supervision on one task lifecycle, with the predictor
    chosen by the configured mode: a cached frozen-trunk probe (``tune_probe``,
    flat or layerwise), a trunk fine-tune (``finetune``), or the pretrained model
    evaluated as loaded (``evaluate_as_is``)."""

    name = "frozen_probe"
    requires_single_process = False

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def prepare(self, context: RunContext) -> Any:
        return prepare_probe_artifacts(
            self.config.steps.prepare,
            context.workspace,
            seed=self.config.seed,
            accelerator=context.accelerator,
            layerwise_config=self.config.steps.layerwise,
            dataset=context.dataset,
            baseline_names=self.config.baselines.names or None,
        )

    def prepare_external(self, context: RunContext, fitted: Any, external: Any) -> Any:
        # Called with the main view's context: external preparation takes the main
        # workspace's configuration (its baseline context), as the frozen probe
        # does for every external set.
        return prepare_external_probe_artifacts(
            self.config.steps.prepare,
            context.workspace,
            fitted,
            external,
            accelerator=context.accelerator,
            seed=self.config.seed,
            layerwise_config=self.config.steps.layerwise,
        )

    def predictor(self, context: RunContext) -> Predictor:
        mode = self.config.steps.mode
        if mode == "evaluate_as_is":
            return PretrainedPredictor(
                predict=self.config.steps.predict, seed=self.config.seed
            )
        if mode == "finetune":
            return FinetunedProbePredictor(
                tune=self.config.steps.tune,
                predict=self.config.steps.predict,
                seed=self.config.seed,
            )
        predictor_cls = (
            LayerwiseCachedProbePredictor
            if self.config.steps.layerwise.enabled
            else CachedProbePredictor
        )
        return predictor_cls(
            tune=self.config.steps.tune,
            predict=self.config.steps.predict,
            seed=self.config.seed,
        )

    def finalize_view(
        self,
        context: RunContext,
        predictions: tuple[PredictionBundle, ...],
        scores: tuple[ScoringResult, ...],
    ) -> None:
        """Paired cross-layer comparison on the view's rows (layerwise, bootstrap on)."""
        if len(predictions) < 2 or not self.config.steps.score.bootstrap_enabled:
            return
        layer_predictions = {
            bundle.payload.metadata.layer_index: bundle.payload
            for bundle in predictions
        }
        comparison_path = compare_layers(
            self.config.steps.score,
            context.workspace,
            layer_predictions,
            seed=self.config.seed,
        )
        if comparison_path is not None:
            logger.info("Paired cross-layer comparison written to %s", comparison_path)

    def score(self, context: RunContext, bundle: PredictionBundle) -> ScoringResult:
        return score_bundle(self.config.steps.score, bundle, seed=self.config.seed)
