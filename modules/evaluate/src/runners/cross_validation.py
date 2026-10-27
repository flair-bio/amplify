"""Cross-validation runner: plan folds, run the shared pipeline per fold,
then pool out-of-fold (OOF) predictions and score them.

Outputs (``<variant>-cv_<cv_id>-run_<config_hash>`` is the experiment)::

    runs/<variant>-cv_<cv_id>-run_<config_hash>/cv_manifest.json
    runs/<variant>-cv_<cv_id>-run_<config_hash>/assignments.parquet
    runs/<variant>-cv_<cv_id>-run_<config_hash>/scores/scores_long.parquet
    runs/<variant>-cv_<cv_id>-run_<config_hash>-fold_<k>/ ordinary per-fold run
    runs/<variant>-cv_<cv_id>-run_<config_hash>-oof/ pooled OOF predictions/scores

A rerun with the same fold plan and config resumes an interrupted experiment.
Different configs, including seeds and hyperparameters, get separate paths;
the fold-plan ``cv_id`` remains shared when the assignments are unchanged.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.dataloader import load_raw_evaluation_dataset
from modules.evaluate.src.dataset.folds import FoldPlan, make_fold_plan
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    get_evaluation_workspace,
)
from modules.evaluate.src.runners import pipeline
from modules.evaluate.src.schemas.artifacts import (
    CrossValidationIdentity,
    EvaluationMetadata,
    PredictOutput,
    ScoreOutput,
)
from modules.evaluate.src.steps.predict import persist_predictions
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.bootstrap import collect_control_arrays, full_scores
from modules.evaluate.src.utils.io import (
    write_json,
    write_parquet,
)
from modules.evaluate.src.utils.long_format import build_long_df
from modules.evaluate.src.utils.seed import seed_everything

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "cv_manifest.json"
ASSIGNMENTS_FILENAME = "assignments.parquet"


def validate_prediction_coverage(
    predictions: pl.DataFrame, expected_ids: Collection[Any], context: str
) -> None:
    """Require exactly one prediction per expected example ID."""
    if Counter(predictions["id"].to_list()) != Counter(expected_ids):
        raise ValueError(
            f"{context} predictions must cover each held-out example exactly once."
        )


def summarize_fold_scores(fold_scores: list[dict[str, float]]) -> dict[str, Any]:
    """Descriptive mean and sample std per metric over folds where it is defined."""
    summary: dict[str, Any] = {}
    for metric in fold_scores[0] if fold_scores else []:
        values = [
            None if np.isnan(scores[metric]) else float(scores[metric])
            for scores in fold_scores
        ]
        valid = [value for value in values if value is not None]
        summary[metric] = {
            "values": values,
            "valid_folds": len(valid),
            "mean": float(np.mean(valid)) if valid else None,
            "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else None,
        }
    return summary


def score_controls(
    output: PredictOutput, task_type: str, metrics: list[str], average: str, seed: int
) -> dict[str, dict[str, float]]:
    """Point-estimate metrics for each requested control condition."""
    task = get_task(task_type)
    frame = output.predictions
    arrays = collect_control_arrays(
        output,
        frame["prediction"].to_list(),
        frame["label"].to_list(),
        frame["probability"].to_list() if "probability" in frame.columns else None,
        np.random.default_rng(seed),
        task,
    )
    return {
        name: full_scores(task, preds, labels, probs, metrics, average)
        for name, (preds, labels, probs) in arrays.items()
    }


def pool_fold_predictions(
    outputs: list[PredictOutput], identity: CrossValidationIdentity, variant: str
) -> PredictOutput:
    """Concatenate fold predictions and row-aligned controls, tagging ``fold_id``."""

    def concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
        return pl.concat(
            [
                frame.with_columns(pl.lit(output.metadata.fold_id).alias("fold_id"))
                for frame, output in zip(frames, outputs)
            ],
            how="vertical_relaxed",
        )

    predictions = concat([output.predictions for output in outputs])
    controls = {
        name: concat([output.controls[name] for output in outputs])
        for name in outputs[0].controls
    }
    for name, frame in controls.items():
        if frame["id"].to_list() != predictions["id"].to_list():
            raise ValueError(f"OOF control '{name}' is not row-aligned to predictions.")
    metadata = outputs[0].metadata.model_copy(
        update={**identity.model_dump(), "variant": variant}
    )
    return PredictOutput(
        metadata=metadata,
        predictions=predictions,
        controls=controls,
        requested_controls=outputs[0].requested_controls,
        config=outputs[0].config,
    )


def _long_rows(
    metadata: EvaluationMetadata | None, scores: dict[str, float]
) -> pl.DataFrame:
    """This run's own point-score rows (ScoreOutput.long_df is the shared table)."""
    return build_long_df(metadata, scores, None, None)


def _config_hash(config: EvaluationConfig) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _completed_folds(
    manifest_path: Path, fingerprint: str, config_hash: str
) -> dict[int, dict[str, Any]]:
    """Fold entries reusable from an interrupted run of the same plan and config."""
    if not manifest_path.exists():
        return {}
    previous = json.loads(manifest_path.read_text())
    if previous.get("status") == "complete":
        raise FileExistsError(f"Cross-validation already complete: {manifest_path}")
    if (previous.get("fingerprint"), previous.get("config_hash")) != (
        fingerprint,
        config_hash,
    ):
        return {}
    return {fold["fold_id"]: fold for fold in previous.get("folds", [])}


def _load_fold(
    workspace: EvaluationWorkspace, cv_id: str, fold_id: int, expected_ids: Collection
) -> PredictOutput | None:
    """Reload a completed fold's predictions, or ``None`` if they're unusable."""
    try:
        output = ScoreStep.load_predict_output(workspace)
        validate_prediction_coverage(
            output.predictions, expected_ids, context=f"Fold {fold_id}"
        )
    except (FileNotFoundError, ValueError, KeyError) as error:
        logger.warning("Rerunning fold %d: %s", fold_id, error)
        return None
    if (output.metadata.cv_id, output.metadata.fold_id) != (cv_id, fold_id):
        return None
    return output


def _fold_counts(plan: FoldPlan, fold_index: int, group_column: str | None) -> dict:
    counts: dict[str, int] = {}
    for role, indices in plan.folds[fold_index].roles().items():
        counts[f"{role}_count"] = len(indices)
        if group_column is not None:
            counts[f"{role}_groups"] = len(set(plan.pool.select(indices)[group_column]))
    return counts


def _release_fold_state(accelerator: Any) -> None:
    """Drop Accelerate references so no fold's model or optimizer persists."""
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_cross_validation(config: EvaluationConfig, accelerator: Any) -> Path:
    """Run every fold, then pooled OOF scoring; return the CV manifest path.

    The manifest stays ``running`` unless every fold and the OOF scoring
    complete; a rerun reuses folds finished under the same plan and config.
    """
    if accelerator.num_processes != 1:
        raise ValueError("Cross-validation currently supports a single process only.")
    if config.external_test_sets:
        logger.warning(
            "Ignoring external_test_sets during cross-validation; they are "
            "scored only by ordinary evaluation runs."
        )
        config = config.model_copy(update={"external_test_sets": []})
    cv = config.cross_validation
    prepare = config.steps.prepare
    ws_config = config.workspace

    plan = make_fold_plan(
        load_raw_evaluation_dataset(
            ws_config.dataset_repo_id or ws_config.dataset_name
        ),
        cv,
        prepare.split,
        task_type=ws_config.task_type,
        sequence_column=prepare.sequence_column,
        label_column=prepare.label_column,
        id_column=prepare.id_column,
    )
    config_hash = _config_hash(config)
    experiment = get_evaluation_workspace(
        ws_config, variant=f"{config.variant}-cv_{plan.cv_id}-run_{config_hash}"
    )
    root = experiment.run_manifest_path.parent
    manifest_path = root / MANIFEST_FILENAME
    completed = _completed_folds(manifest_path, plan.fingerprint, config_hash)

    manifest: dict[str, Any] = {
        "status": "running",
        "cv_id": plan.cv_id,
        "fingerprint": plan.fingerprint,
        "config_hash": config_hash,
        "seed": config.seed,
        "cross_validation": cv.model_dump(mode="json"),
        "split": prepare.split.model_dump(mode="json"),
        "assignments": str(
            write_parquet(plan.assignments, root / ASSIGNMENTS_FILENAME)
        ),
        "folds": [],
    }
    write_json(manifest_path, manifest)

    fold_predictions: list[PredictOutput] = []
    long_frames: list[pl.DataFrame] = []
    for index, fold in enumerate(plan.folds):
        progress = f"{index + 1}/{len(plan.folds)}"
        workspace = get_evaluation_workspace(
            ws_config, variant=f"{experiment.variant}-fold_{fold.fold_id}"
        )
        expected_ids = plan.pool.select(fold.test)[prepare.id_column]
        entry = completed.get(fold.fold_id)
        prediction = (
            _load_fold(workspace, plan.cv_id, fold.fold_id, expected_ids)
            if entry is not None
            else None
        )
        if entry is not None and prediction is not None:
            logger.info("=== Cross-validation fold %s: reused ===", progress)
            scores = entry["scores"]
        else:
            logger.info("=== Cross-validation fold %s ===", progress)
            # Same training seed per fold; split randomness lives in split_seed.
            seed_everything(config.seed)
            result = pipeline.run_pipeline(
                config,
                workspace,
                accelerator,
                dataset=fold.dataset(plan.pool),
                cross_validation=CrossValidationIdentity(
                    cv_id=plan.cv_id, fold_id=fold.fold_id, split_seed=cv.split_seed
                ),
            )
            prediction = result.predictions[None]
            scored = result.scores[None]
            assert isinstance(scored, ScoreOutput)
            scores = scored.scores
            validate_prediction_coverage(
                prediction.predictions, expected_ids, context=f"Fold {fold.fold_id}"
            )
            del result
            _release_fold_state(accelerator)
        fold_predictions.append(prediction)
        long_frames.append(_long_rows(prediction.metadata, scores))
        manifest["folds"].append(
            {
                "fold_id": fold.fold_id,
                "variant": workspace.variant,
                "run_manifest": str(workspace.run_manifest_path),
                "seed": config.seed,
                **_fold_counts(plan, index, cv.group_column),
                "scores": scores,
            }
        )
        write_json(manifest_path, manifest)

    oof_workspace = get_evaluation_workspace(
        ws_config, variant=f"{experiment.variant}-oof"
    )
    oof_workspace.setup_directories()
    oof = pool_fold_predictions(
        fold_predictions,
        CrossValidationIdentity(cv_id=plan.cv_id, split_seed=cv.split_seed),
        oof_workspace.variant or "",
    )
    validate_prediction_coverage(
        oof.predictions, plan.pool[prepare.id_column], context="OOF"
    )
    persist_predictions(oof_workspace, oof, config.seed)

    logger.info("=== Scoring pooled out-of-fold predictions ===")
    oof_score = ScoreStep(config.steps.score).run(
        oof_workspace, seed=config.seed, predict_result=oof
    )
    long_frames.append(_long_rows(oof_score.metadata, oof_score.scores))
    long_path = write_parquet(
        pl.concat(long_frames, how="diagonal_relaxed"),
        experiment.scores_path / config.steps.score.long_filename,
    )

    metrics = list(oof_score.scores)
    manifest.update(
        status="complete",
        fold_summary=summarize_fold_scores([f["scores"] for f in manifest["folds"]]),
        oof_predictions=str(oof.predictions_path),
        oof_scores=oof_score.scores,
        control_oof_scores=score_controls(
            oof,
            ws_config.task_type,
            metrics,
            config.steps.score.metric_aggregation,
            config.seed,
        ),
        scores_long=str(long_path),
    )
    write_json(manifest_path, manifest)
    return manifest_path
