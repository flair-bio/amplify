"""Attach configured baseline predictors to a prediction result.

Baselines are task-owned, non-trained predictors (for example majority-class
or train-mean constants, or external prediction files). Their artifacts are
loaded and aligned to the trained predictions here, so scoring consumes them
exactly like model controls. Loading primitives live in
:mod:`modules.evaluate.src.utils.baselines`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from modules.evaluate.src.schemas.artifacts import PredictOutput
from modules.evaluate.src.utils.baselines import (
    align_baseline,
    artifact_identity,
    load_baseline_artifact,
)


def attach_baselines(
    predict_result: PredictOutput, regression: bool, names: list[str]
) -> tuple[PredictOutput, dict[str, dict[str, Any]]]:
    """Return *predict_result* with baseline controls added, plus their references.

    The reference record for each baseline holds the artifact path and its
    identity; the caller adds per-condition scores before publication.
    Raises if a baseline name collides with a model control.
    """
    baseline_references: dict[str, dict[str, Any]] = {}
    if not names:
        return predict_result, baseline_references
    if not predict_result.baseline_context:
        raise ValueError(
            "Saved predictions lack dataset/split/label provenance (baseline_context). "
            "Recover and validate the original evaluation context before comparison; "
            "standalone scoring does not require baselines."
        )

    context = predict_result.baseline_context
    predictions_df = predict_result.predictions
    controls = dict(predict_result.controls)
    requested_controls = list(predict_result.requested_controls)
    if context["fit"]["task_type"] != predict_result.metadata.task_type:
        raise ValueError("Baseline context task does not match model predictions.")
    if (
        predict_result.metadata.dataset_repo_id is not None
        and context["test"]["dataset"]["source"]
        != predict_result.metadata.dataset_repo_id
    ):
        raise ValueError("Baseline context dataset does not match model predictions.")
    if len(set(names)) != len(names):
        raise ValueError("Comparison baseline names must be unique.")
    from modules.evaluate.src.utils.baselines import BaselinesConfig

    BaselinesConfig(names=names).validate_task(predict_result.metadata.task_type)
    for name in names:
        if name in controls:
            raise ValueError(f"Baseline '{name}' is also a model control.")
        frame, path = load_baseline_artifact(context, name)
        controls[name] = align_baseline(
            frame,
            predictions_df,
            regression=regression,
        )
        requested_controls.append(name)
        baseline_references[name] = {
            "path": str(path),
            "identity": artifact_identity(context, name),
        }
    attached = replace(
        predict_result, controls=controls, requested_controls=requested_controls
    )
    return attached, baseline_references
