"""Prediction artifacts: persist a :class:`PredictOutput` and reload it from disk.

The run manifest's ``predict`` section is the contract between the predict
and score stages: it records the metadata, config, control cache paths and
baseline context needed to rerun scoring without the in-memory result.
"""

from __future__ import annotations

import hashlib
import json
import logging
from copy import copy
from pathlib import Path
from time import monotonic
from typing import Any

import polars as pl

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.schemas.artifacts import (
    ContactEvaluationMetadata,
    EvaluationMetadata,
    PredictOutput,
)
from modules.evaluate.src.utils.io import (
    downcast_prediction_dtypes,
    update_run_manifest,
    write_parquet,
)

logger = logging.getLogger(__name__)


def record_token_positions(
    frame: pl.DataFrame, context: dict[str, Any] | None
) -> pl.DataFrame:
    """Carry the original tokenization population through full-row or cached inference."""
    if (
        "evaluated_positions" in frame.columns
        or not context
        or "population" not in context
    ):
        return frame
    population = {json.dumps(row["id"]): row for row in context["population"]}
    if len(population) != len(context["population"]):
        logger.warning(
            "Cannot record unambiguous residue positions with repeated source IDs."
        )
        return frame
    offsets: dict[str, int] = {}
    positions = []
    for row in frame.iter_rows(named=True):
        key = json.dumps(row["id"])
        source = population.get(key)
        offset = offsets.get(key, 0)
        length = len(row["label"])
        if source is None or row["label"] != source["label"][offset : offset + length]:
            logger.warning(
                "Cannot recover evaluated residue positions from the saved population."
            )
            return frame
        positions.append(source["evaluated_positions"][offset : offset + length])
        offsets[key] = offset + length
    return frame.with_columns(
        pl.Series("evaluated_positions", positions, dtype=pl.List(pl.Int64))
    )


def output_workspace(
    workspace: EvaluationWorkspace, directory: Path
) -> EvaluationWorkspace:
    """Use a separate artifact directory without changing input identity."""
    result = copy(workspace)
    for name in (
        "run_root",
        "preds_path",
        "stats_path",
        "scores_path",
        "comparison_path",
    ):
        setattr(result, name, directory)
    result.run_manifest_path = directory / "run_manifest.json"
    result._all_paths = [directory]
    return result


def record_prediction_population(output: PredictOutput) -> None:
    output.predictions = record_token_positions(
        output.predictions, output.baseline_context
    )
    if "evaluated_positions" not in output.predictions.columns:
        return
    for name, frame in output.controls.items():
        if (
            frame["id"].to_list() == output.predictions["id"].to_list()
            and frame["prediction"].list.len().to_list()
            == output.predictions["prediction"].list.len().to_list()
        ):
            output.controls[name] = frame.with_columns(
                output.predictions["evaluated_positions"]
            )


def persist_predictions(
    workspace: EvaluationWorkspace, output: PredictOutput, seed: int
) -> None:
    """Write predictions and uncached controls, then record in the run
    manifest everything scoring needs to rerun as a disk-only process."""
    started = monotonic()
    logger.info("Persisting predictions started (%s)", workspace.preds_path)
    config = output.config or {}
    record_prediction_population(output)
    output.predictions_path = write_parquet(
        downcast_prediction_dtypes(output.predictions),
        workspace.preds_path
        / config.get("predictions_filename", "predictions.parquet"),
    )
    for name, frame in output.controls.items():
        if name not in output.control_paths or "evaluated_positions" in frame.columns:
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
            "workspace": workspace.config.model_dump(mode="json"),
            "predictions_path": str(output.predictions_path.resolve()),
            "predictions_sha256": hashlib.sha256(
                output.predictions_path.read_bytes()
            ).hexdigest(),
        },
    )
    logger.info(
        "Persisting predictions finished in %.1fs (%d rows, %d controls)",
        monotonic() - started,
        output.predictions.height,
        len(output.controls),
    )


def load_predict_output(workspace: EvaluationWorkspace) -> PredictOutput:
    """Reconstruct a :class:`PredictOutput` purely from disk: the run
    manifest's "predict" section plus the predictions/controls parquet files
    it points to. This is what lets scoring run as a plain, disk-only step
    with no in-memory ``PredictOutput``/``Accelerator`` dependency."""
    if not workspace.run_manifest_path.exists():
        raise FileNotFoundError(
            f"No run_manifest.json found at {workspace.run_manifest_path}; "
            "run the Predict step before scoring."
        )
    manifest = json.loads(workspace.run_manifest_path.read_text())
    predict_section = manifest.get("predict")
    if predict_section is None:
        raise ValueError(
            f"{workspace.run_manifest_path} has no 'predict' section; "
            "run the Predict step before scoring."
        )

    config_dict = predict_section["config"]
    predictions_path = workspace.preds_path / config_dict.get(
        "predictions_filename", "predictions.parquet"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found at {predictions_path}; run the "
            "Predict step before scoring."
        )
    checksum = predict_section.get("predictions_sha256")
    if (
        checksum is not None
        and hashlib.sha256(predictions_path.read_bytes()).hexdigest() != checksum
    ):
        raise ValueError(f"Corrupt saved model predictions: {predictions_path}")

    metadata_dict = predict_section.get("metadata")
    control_paths = {
        control: Path(path)
        for control, path in predict_section.get("control_paths", {}).items()
    }
    return PredictOutput(
        metadata=(
            ContactEvaluationMetadata
            if "protocol_version" in metadata_dict
            else EvaluationMetadata
        ).model_validate(metadata_dict),
        predictions=pl.read_parquet(predictions_path),
        predictions_path=predictions_path,
        controls={
            control: pl.read_parquet(path) for control, path in control_paths.items()
        },
        control_paths=control_paths,
        requested_controls=list(config_dict.get("controls", [])),
        config=config_dict,
        baseline_context=predict_section.get("baseline_context"),
    )
