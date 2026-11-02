"""
Compute task-specific metrics for the trained model's predictions
(:class:`~modules.evaluate.src.steps.predict.PredictStep`'s output), and,
optionally, bootstrap confidence intervals for those metrics plus a
statistical comparison between the trained model and each configured control
condition (untrained/randomly re-initialized model variants and scrambled
labels/sequences) to quantify how much of its performance is attributable to
learning from the data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl

from modules.evaluate.src.contracts.artifacts import (
    PredictOutput,
    ScoreOutput,
    StructuredScoreOutput,
)
from modules.evaluate.src.contracts.config import ScoreConfig
from modules.evaluate.src.dataset.baseline_data import (
    align_baseline,
    align_token_baseline,
    artifact_identity,
    identity_hash,
    load_baseline_artifact,
    mmseqs_coverage,
)
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.metrics.scoring import compare_layer_scores
from modules.evaluate.src.metrics.tables import upsert_long_df
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.registry import get_flat_task
from modules.evaluate.src.utils.io import (
    update_run_manifest,
    write_json,
    write_parquet,
    write_settings_json,
)
from modules.evaluate.src.utils.tracking import log_wandb

logger = logging.getLogger(__name__)

LAYERWISE_COMPARISON_FILENAME = "layerwise_comparison.parquet"


class ScoreStep:
    def __init__(self, config: ScoreConfig) -> None:
        self.config = config

    def run(
        self,
        workspace: EvaluationWorkspace,
        seed: int = 710019,
        predict_result: PredictOutput | None = None,
        *,
        include_shared_baselines: bool = False,
    ) -> ScoreOutput | StructuredScoreOutput:
        """Load predictions, invoke the task's scoring protocol, and publish."""
        logger.info("Running Score step...")
        if include_shared_baselines:
            raise ValueError(
                "Shared baseline comparisons are disk-only; use compare_saved()."
            )
        if predict_result is None:
            predict_result = self.load_predict_output(workspace)
        task = get_task(predict_result.metadata.task_type, workspace.num_labels)
        result = task.score(predict_result, self.config, seed=seed)
        if isinstance(result, StructuredScoreOutput):
            return self._publish_structured_scores(
                result=result,
                output_dir=workspace.scores_path,
                manifest_path=workspace.run_manifest_path,
                baseline_references={},
            )
        return self._publish_flat_scores(workspace, result, {}, seed=seed)

    def compare_saved(
        self,
        workspace: EvaluationWorkspace,
        *,
        shared_baselines: Sequence[str],
        controls: Sequence[str],
        seed: int = 710019,
        expected_preprocessing: dict[str, Any] | None = None,
    ) -> Path:
        """Compare saved predictions and selected artifacts without touching scores."""
        predictions = self.load_predict_output(workspace)
        _validate_prediction_workspace(workspace, predictions)
        if not shared_baselines and not controls:
            raise ValueError(
                "Select at least one saved control or shared baseline to compare."
            )
        if shared_baselines and predictions.baseline_context is None:
            raise ValueError(
                "Saved predictions have no baseline provenance. Re-run evaluation "
                "with current provenance support before comparing shared baselines."
            )
        baseline_context = predictions.baseline_context
        if baseline_context is None:
            baseline_context = {}
        if expected_preprocessing is not None and predictions.baseline_context:
            recorded = predictions.baseline_context.get("test", {}).get("preprocessing")
            if recorded != expected_preprocessing:
                raise ValueError(
                    "Comparison preprocessing does not match the saved prediction "
                    "population. Use the same split, columns, crop policy, and "
                    "max_length used for evaluation."
                )
        saved_controls = set(predictions.requested_controls)
        unavailable = set(controls) - saved_controls
        if unavailable:
            raise ValueError(
                f"Requested controls {sorted(unavailable)} were not saved with "
                "these predictions; comparison cannot run inference."
            )
        if set(shared_baselines) & set(controls):
            raise ValueError("A baseline cannot also be selected as a saved control.")

        condition_frames = dict(predictions.controls)
        references: dict[str, dict[str, Any]] = {}
        task = get_task(predictions.metadata.task_type, workspace.num_labels)
        for name in shared_baselines:
            frame, path = load_baseline_artifact(baseline_context, name)
            if task.name == "contact_prediction":
                if name != "separation_prior":
                    raise ValueError(f"Unsupported shared contact baseline: {name}.")
                condition_frames[name] = _align_contact_baseline(
                    frame, predictions.predictions, name
                )
            elif task.name == "token_classification":
                condition_frames[name] = align_token_baseline(
                    frame, predictions.predictions
                )
            else:
                condition_frames[name] = align_baseline(
                    frame,
                    predictions.predictions,
                    regression="continuous" in task.accepted_target_types,
                    match_labels=task.name != "pseudo_perplexity",
                )
            references[name] = {
                "path": str(path),
                "identity": artifact_identity(baseline_context, name),
            }
            if name == "mmseqs_knn":
                references[name]["coverage"] = mmseqs_coverage(frame)

        predictions = replace(
            predictions,
            controls={
                name: condition_frames[name]
                for name in (*controls, *shared_baselines)
                if name in condition_frames
            },
            requested_controls=[*controls, *shared_baselines],
        )
        result = task.score(predictions, self.config, seed=seed)
        output_dir = workspace.run_root / "comparisons" / "shared_baselines"
        selection = {
            "baselines": sorted(shared_baselines),
            "controls": sorted(controls),
            "score_config": self.config.model_dump(mode="json"),
        }
        output_dir = output_dir / f"comparison-{identity_hash(selection)[:12]}"
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = predictions.metadata.model_dump(mode="json")
        if isinstance(result, StructuredScoreOutput):
            write_parquet(result.unit_scores, output_dir / "unit_scores.parquet")
            write_parquet(result.summaries, output_dir / "scores.parquet")
            for name, frame in result.control_unit_scores.items():
                write_parquet(
                    frame,
                    output_dir / "controls" / name / "unit_scores.parquet",
                )
                write_parquet(
                    result.control_summaries[name],
                    output_dir / "controls" / name / "scores.parquet",
                )
            payload = {
                "metadata": metadata,
                "config": self.config.model_dump(mode="json"),
                "baselines": references,
                "scores": result.summaries.to_dicts(),
                "controls": {
                    name: summary.to_dicts()
                    for name, summary in result.control_summaries.items()
                },
            }
        else:
            if result.long_df is not None:
                write_parquet(result.long_df, output_dir / "comparison_long.parquet")
            if result.resamples is not None:
                write_parquet(
                    result.resamples, output_dir / self.config.output_filename
                )
            payload = {
                "metadata": metadata,
                "config": result.config,
                "baselines": references,
                "condition_scores": result.condition_scores,
                "control_comparisons": result.control_comparisons or {},
                "confidence_intervals": result.confidence_intervals or {},
            }
        path = write_json(output_dir / "comparison.json", payload)
        return path

    def _publish_flat_scores(
        self,
        workspace: EvaluationWorkspace,
        result: ScoreOutput,
        baseline_references: dict[str, dict[str, Any]],
        *,
        seed: int,
    ) -> ScoreOutput:
        metadata_dict = (
            result.metadata.model_dump(mode="json") if result.metadata else {}
        )
        layer_index = result.metadata.layer_index if result.metadata else None
        wandb_prefix = "test" if layer_index is None else f"test/layer_{layer_index}"
        log_wandb(
            {f"{wandb_prefix}/{name}": value for name, value in result.scores.items()}
        )
        for name, reference in baseline_references.items():
            reference["scores"] = result.condition_scores[name]
        result.scores_path = update_run_manifest(
            workspace.run_manifest_path,
            "score",
            {
                "metadata": metadata_dict,
                "config": result.config,
                "scores": result.scores,
                "baselines": baseline_references,
                "condition_scores": result.condition_scores,
            },
        )
        if result.resamples is not None:
            result.output_path = write_parquet(
                result.resamples, workspace.stats_path / self.config.output_filename
            )
            result.summary_path = update_run_manifest(
                workspace.run_manifest_path,
                "bootstrap",
                {
                    "seed": seed,
                    "config": result.config,
                    "metadata": metadata_dict,
                    "confidence_intervals": result.confidence_intervals,
                    "control_comparisons": result.control_comparisons or {},
                },
            )
            logger.info("Bootstrap summary written to %s", result.summary_path)
        long_path = workspace.comparison_path / self.config.long_filename
        variant = result.metadata.variant if result.metadata else None
        split = result.metadata.split if result.metadata else None
        assert result.long_df is not None
        result.long_df = upsert_long_df(long_path, result.long_df, variant, split)
        result.long_path = write_parquet(result.long_df, long_path)
        return result

    def compare_layers(
        self,
        workspace: EvaluationWorkspace,
        layer_predictions: dict[int, PredictOutput],
        seed: int = 710019,
    ) -> Path | None:
        """Publish paired comparisons with the preselected deepest layer."""
        comparison = compare_layer_scores(
            get_flat_task(workspace.task_type, workspace.num_labels),
            self.config,
            layer_predictions,
            variant=workspace.variant,
            seed=seed,
        )
        if comparison is None:
            return None
        path = workspace.comparison_path / LAYERWISE_COMPARISON_FILENAME
        return write_parquet(
            upsert_long_df(
                path,
                comparison,
                workspace.variant,
                layer_predictions[max(layer_predictions)].metadata.split,
            ),
            path,
        )

    @staticmethod
    def load_predict_output(workspace: EvaluationWorkspace) -> PredictOutput:
        """Reconstruct a :class:`PredictOutput` purely from disk: the run
        manifest's "predict" section (metadata/config/control cache paths)
        plus the predictions/controls parquet files it points to. This is
        what lets :meth:`run` above be a plain, disk-only step with no
        in-memory ``PredictOutput``/``Accelerator`` dependency."""
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
        predictions_path = workspace.preds_path / config_dict["predictions_filename"]
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions file not found at {predictions_path}; run the "
                "Predict step before scoring."
            )

        metadata_dict = predict_section.get("metadata")
        control_paths = {
            control: Path(path)
            for control, path in predict_section.get("control_paths", {}).items()
        }
        missing_controls = {
            control: path
            for control, path in control_paths.items()
            if not path.exists()
        }
        if missing_controls:
            raise FileNotFoundError(
                "Saved control predictions are missing: "
                + ", ".join(f"{name}={path}" for name, path in missing_controls.items())
            )
        return PredictOutput(
            metadata=get_task(workspace.task_type).prediction_metadata(metadata_dict),
            predictions=pl.read_parquet(predictions_path),
            predictions_path=predictions_path,
            controls={
                control: pl.read_parquet(path)
                for control, path in control_paths.items()
            },
            control_paths=control_paths,
            requested_controls=[
                *config_dict.get("controls", []),
                *config_dict.get("baselines", []),
            ],
            config=config_dict,
            baseline_context=predict_section.get("baseline_context"),
        )

    def _publish_structured_scores(
        self,
        *,
        result: StructuredScoreOutput,
        output_dir: Path,
        manifest_path: Path,
        baseline_references: dict[str, dict[str, Any]],
    ) -> StructuredScoreOutput:
        logger.info("Scores: %s", result.summaries)
        seed = result.metadata.seed
        if seed is None:
            raise ValueError("Scoring requires an explicit run seed")
        result.unit_scores_path = write_parquet(
            result.unit_scores, output_dir / "unit_scores.parquet"
        )
        result.summaries_path = write_parquet(
            result.summaries, output_dir / "scores.parquet"
        )
        for name, frame in result.control_unit_scores.items():
            control_dir = output_dir / "controls" / name
            result.control_unit_scores_paths[name] = write_parquet(
                frame, control_dir / "unit_scores.parquet"
            )
            result.control_summaries_paths[name] = write_parquet(
                result.control_summaries[name],
                control_dir / "scores.parquet",
            )
        for name, reference in baseline_references.items():
            if name in result.control_summaries:
                reference["scores"] = result.control_summaries[name].to_dicts()
                reference["scores_path"] = str(result.control_summaries_paths[name])
                reference["unit_scores_path"] = str(
                    result.control_unit_scores_paths[name]
                )
        write_settings_json(
            output_dir / result.settings_filename,
            seed,
            self.config.model_dump(),
            metadata=result.metadata.model_dump(mode="json"),
        )
        update_run_manifest(
            manifest_path,
            "score",
            {
                "metadata": result.metadata.model_dump(mode="json"),
                "config": self.config.model_dump(mode="json"),
                "scores": result.summaries.to_dicts(),
                "scores_path": str(result.summaries_path),
                "unit_scores_path": str(result.unit_scores_path),
                "controls": {
                    name: {
                        "scores": frame.to_dicts(),
                        "scores_path": str(result.control_summaries_paths[name]),
                        "unit_scores_path": str(result.control_unit_scores_paths[name]),
                    }
                    for name, frame in result.control_summaries.items()
                },
                "baselines": baseline_references,
            },
        )
        return result


def _align_contact_baseline(
    baseline: pl.DataFrame, predictions: pl.DataFrame, name: str
) -> pl.DataFrame:
    """Require exact candidate/label correspondence before scoring a shared prior."""
    required = {
        "id",
        "record_id",
        "unit_id",
        "split",
        "source",
        "edition",
        "complex_id",
        "target",
        "kind",
        "labels_available",
        "original_i",
        "original_j",
        "labels",
        "length",
        "status",
    }
    if not required.issubset(baseline.columns) or "scores" not in baseline.columns:
        raise ValueError(f"{name} artifact is missing contact prediction columns.")
    if not required.issubset(predictions.columns):
        raise ValueError("Primary contact predictions are missing candidate columns.")

    def keyed_rows(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
        rows = frame.to_dicts()
        keyed = {row["id"]: row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"{name} artifact contains duplicate contact IDs.")
        return keyed

    baseline_rows = keyed_rows(baseline)
    prediction_rows = keyed_rows(predictions)
    if set(baseline_rows) != set(prediction_rows):
        raise ValueError(f"{name} contact rows do not match primary prediction IDs.")
    for identifier, row in baseline_rows.items():
        primary = prediction_rows[identifier]
        if any(row[key] != primary[key] for key in required - {"id"}):
            raise ValueError(
                f"{name} contact candidates differ from primary predictions."
            )
    return baseline


def _validate_prediction_workspace(
    workspace: EvaluationWorkspace, predictions: PredictOutput
) -> None:
    metadata = predictions.metadata
    expected = {
        "model_name": workspace.model_name,
        "dataset_name": workspace.dataset_name,
        "task_type": workspace.task_type,
        "dataset_repo_id": workspace.dataset_repo_id,
        "dataset_config_name": workspace.config.dataset_config_name,
        "dataset_revision": workspace.config.dataset_revision,
    }
    mismatches = [
        key for key, value in expected.items() if getattr(metadata, key) != value
    ]
    if workspace.variant is not None and metadata.variant != workspace.variant:
        mismatches.append("variant")
    if mismatches:
        raise ValueError(
            "Saved prediction provenance does not match the selected workspace "
            f"({', '.join(mismatches)})."
        )
