"""Predict every CASP unit with primary and independently fitted predictors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.evaluate.src.artifacts.results import PredictOutput
from modules.evaluate.src.dataset.contact import ContactDataset
from modules.evaluate.src.model.controls import ControlPredictors
from modules.evaluate.src.model.factory import Predictor
from modules.evaluate.src.tasks.contact_prediction import (
    predict_units,
    supported_controls,
)

if TYPE_CHECKING:
    from modules.evaluate.src.artifacts.results import EvaluationMetadata
    from modules.evaluate.src.config.evaluation import PredictConfig


def predict_contact(
    predictor: Predictor,
    dataset: ContactDataset,
    control_predictors: ControlPredictors,
    metadata: EvaluationMetadata,
    config: PredictConfig,
) -> PredictOutput:
    if dataset is None:
        raise TypeError("Contact observations require contact data.")
    unsupported = set[str](config.controls) - supported_controls
    if unsupported:
        raise ValueError(
            f"CASP contact prediction supports only random_trunk; got {sorted(unsupported)}"
        )
    if config.collect_probabilities:
        raise ValueError("CASP contacts consume pair scores, not class probabilities")
    for name in config.controls:
        if name not in control_predictors.independent_models:
            raise ValueError(
                f"{name} was requested but no fitted contact control is available"
            )
    output = predict_units(
        dataset,
        lambda chain: predictor.model.predict(chain, predictor.tokenizer),
        metadata,
    )
    for name in config.controls:
        control = control_predictors.independent_models[name]
        output.controls[name] = predict_units(
            dataset, lambda chain: control.predict(chain, predictor.tokenizer), metadata
        ).predictions
    output.requested_controls = list(config.controls)
    return output
