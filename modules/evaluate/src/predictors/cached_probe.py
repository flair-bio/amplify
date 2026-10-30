"""Cached frozen-trunk probe predictor: fits and predicts through the training and
predictor owners, with no step or pipeline involvement.

Random-trunk and other predict-side controls are handled inside
``fit_probe_protocol`` (controls) and ``predict_trained``, so this class stays
a thin, configured call into those owners.
"""

from __future__ import annotations

from typing import Any

from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.dataset.workspace import get_layer_workspace
from modules.evaluate.src.evaluation.contracts import (
    FitOutcome,
    FitSummary,
    PredictionBundle,
    RunContext,
)
from modules.evaluate.src.predictors.inference import predict_layerwise, predict_trained
from modules.evaluate.src.training.config import TuneConfig
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.training.tuning import fit_probe_protocol


class CachedProbePredictor:
    """Frozen-trunk MLP/linear probe fitted on cached trunk features."""

    name = "cached_probe"
    transfers_to_external = True
    #: Whether fitting updates the trunk (``steps.mode: finetune``).
    train_trunk = False

    def __init__(self, *, tune: TuneConfig, predict: PredictConfig, seed: int) -> None:
        self.tune = tune
        self.predict_config = predict
        self.seed = seed

    def fit(self, context: RunContext, prepared: Any) -> FitOutcome:
        trainer = TorchProbeTrainer(self.tune, train_trunk=self.train_trunk)
        fitted = fit_probe_protocol(
            trainer,
            context.workspace,
            prepared,
            seed=self.seed,
            accelerator=context.accelerator,
            controls=self.predict_config.controls,
        )
        summary = FitSummary(
            predictor=self.name,
            details={"head_type": self.tune.head.head_type},
        )
        return FitOutcome(summary=summary, fitted=fitted)

    def predict(
        self, context: RunContext, prepared: Any
    ) -> tuple[PredictionBundle, ...]:
        output = predict_trained(
            context.workspace,
            prepared,
            self.predict_config,
            seed=self.seed,
            head_type=self.tune.head.head_type,
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


class LayerwiseCachedProbePredictor(CachedProbePredictor):
    """Layerwise probes over one cached trunk: one fit, one prediction branch per layer.

    Fitting is the ordinary cached-probe fit (``fit_probe_protocol`` dispatches
    the layerwise heads). Each probed layer yields its own branch whose workspace
    holds that layer's predictions, controls and scores.
    """

    name = "cached_probe_layerwise"

    def predict(
        self, context: RunContext, prepared: Any
    ) -> tuple[PredictionBundle, ...]:
        outputs = predict_layerwise(
            context.workspace,
            prepared,
            self.predict_config,
            context.accelerator,
            self.seed,
        )
        return tuple(
            PredictionBundle(
                predictor=self.name,
                split=context.split,
                branch=f"layer_{layer}",
                workspace=get_layer_workspace(context.workspace, layer),
                payload=output,
            )
            for layer, output in outputs.items()
        )


class FinetunedProbePredictor(CachedProbePredictor):
    """Trunk fine-tuning with an MLP head on live (uncached) inputs.

    Same fit protocol and predict path as the cached probe; the only difference
    is that the trainer updates the trunk, so no embedding cache is involved.
    """

    name = "finetuned_probe"
    transfers_to_external = True
    train_trunk = True
