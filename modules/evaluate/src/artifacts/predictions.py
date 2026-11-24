"""Prediction and condition artifact publication."""

from __future__ import annotations

import logging
from time import monotonic

from modules.evaluate.src.artifacts.files import (
    downcast_prediction_dtypes,
    write_parquet,
)
from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.results import PredictOutput
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
)

logger = logging.getLogger(__name__)


def persist_predictions(
    workspace: EvaluationWorkspace, output: PredictOutput, seed: int
) -> None:
    """Write predictions, uncached controls and their reproducibility metadata."""
    started = monotonic()
    logger.info("Persisting predictions started (%s)", workspace.preds_path)
    config = output.config or {}
    output.predictions_path = write_parquet(
        downcast_prediction_dtypes(output.predictions),
        workspace.preds_path
        / config.get("predictions_filename", "predictions.parquet"),
    )
    for name, frame in output.controls.items():
        if name not in output.control_paths:
            output.control_paths[name] = write_parquet(
                downcast_prediction_dtypes(frame),
                workspace.control_workspace(name).preds_path / "predictions.parquet",
            )
    update_run_manifest(
        workspace.run_manifest_path,
        "predict",
        {
            "seed": seed,
            "config": config,
            "metadata": output.metadata.model_dump(mode="json"),
            "control_paths": {k: str(v) for k, v in output.control_paths.items()},
            "baseline_context": output.baseline_context,
        },
        namespace=workspace.manifest_namespace,
    )
    logger.info(
        "Persisting predictions finished in %.1fs (%d rows, %d controls)",
        monotonic() - started,
        output.predictions.height,
        len(output.controls),
    )
