"""Pretrained masked-LM predictor: evaluates the model as loaded, with no fitting.

Nothing is tuned, so there is no fitted state to transfer to external views.
Predict-side controls (the ``random_trunk`` condition, which re-initializes the
trunk with the same architecture and tokenizer and never touches the primary
model) are produced by ``predict_trained`` through the control builder.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.evaluation.contracts import (
    FitOutcome,
    FitSummary,
    PredictionBundle,
    RunContext,
)
from modules.evaluate.src.predictors.inference import predict_trained
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts


class PretrainedPredictor:
    """Predicts with the pretrained model; the fit is a device placement only."""

    name = "pretrained"
    transfers_to_external = False

    def __init__(self, *, predict: PredictConfig, seed: int) -> None:
        self.predict_config = predict
        self.seed = seed

    def fit(self, context: RunContext, prepared: Any) -> FitOutcome:
        # Prepare leaves the raw model unwrapped; the predict loop needs it
        # placed by accelerate exactly once, after which nothing else changes.
        assert isinstance(prepared, PreparedArtifacts)
        fitted = replace(prepared, model=context.accelerator.prepare(prepared.model))
        summary = FitSummary(predictor=self.name, details={"fitted": False})
        return FitOutcome(summary=summary, fitted=fitted)

    def predict(
        self, context: RunContext, prepared: Any
    ) -> tuple[PredictionBundle, ...]:
        # No trained head exists, so there is no head_type to report.
        output = predict_trained(
            context.workspace,
            prepared,
            self.predict_config,
            seed=self.seed,
            head_type=None,
            variant=context.workspace.variant,
            accelerator=context.accelerator,
        )
        return (
            PredictionBundle(
                predictor=self.name,
                split=context.split,
                branch="main",
                workspace=context.workspace,
                payload=output,
            ),
        )
