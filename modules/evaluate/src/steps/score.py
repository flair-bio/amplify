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
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.schemas.artifacts import (
    ConfidenceInterval,
    EvaluationMetadata,
    PredictOutput,
    ScoreOutput,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.bootstrap import (
    as_indexable_array,
    ci_stats,
    collect_control_arrays,
    full_scores,
    summarize_controls,
)
from modules.evaluate.src.utils.io import (
    update_run_manifest,
    write_parquet,
)
from modules.evaluate.src.utils.long_format import build_long_df, upsert_long_df
from modules.evaluate.src.utils.tracking import log_wandb

logger = logging.getLogger(__name__)

# Minimum n_samples below which bootstrap confidence intervals and
# trained-vs-control p-values are too coarse/unstable to support a claim of
# statistical significance: with n resamples, the smallest achievable
# nonzero p-value is 1/n, and percentile-based CIs need enough resamples to
# actually populate their tails rather than interpolating from a handful of
# points. 1000 matches standard bootstrap guidance (Efron & Tibshirani).
_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES = 1000

# Permutation p-value resolution is 1/(n+1) regardless of bootstrap n_samples,
# and Benjamini-Hochberg across controls raises the smallest achievable
# adjusted value further, so this needs the same order of magnitude as the CI.
_DEFAULT_N_PERMUTATIONS = 1000


class ScoreConfig(BaseModel):
    """Config. for the score step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] | None = None
    metric_aggregation: Literal["micro", "macro", "weighted"] = "weighted"

    # Bootstrap confidence intervals / control comparisons are optional
    # on top of the (always-computed) point-estimate metrics above. Default
    # n_samples matches _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES rather than an
    # arbitrarily small number, so enabling bootstrap without also setting
    # n_samples doesn't silently produce CIs/p-values too coarse to trust.
    bootstrap_enabled: bool = False
    n_samples: int = Field(_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES, gt=0)
    n_permutations: int = Field(_DEFAULT_N_PERMUTATIONS, gt=0)
    output_filename: str = "bootstrap_results.parquet"
    confidence_interval: float = Field(0.95, gt=0.0, lt=1.0)

    # Tidy/long-format sidecar (see ``build_long_df``), written on every run
    # (with or without bootstrap) so downstream plotting/aggregation code has
    # a single, concatenable artifact per (model, dataset) run regardless of
    # which optional outputs above were enabled.
    long_filename: str = "scores_long.parquet"

    @model_validator(mode="after")
    def _warn_if_resamples_too_low_for_significance(self) -> ScoreConfig:
        if not self.bootstrap_enabled:
            return self
        if self.n_samples < _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES:
            logger.warning(
                "ScoreConfig.n_samples=%d is well below the %d resamples "
                "generally recommended for reliable bootstrap confidence "
                "intervals (percentile intervals are estimated from only %d "
                "points). Do not use these results to support a claim of "
                "statistical significance without increasing n_samples.",
                self.n_samples,
                _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES,
                self.n_samples,
            )
        if self.n_permutations < _DEFAULT_N_PERMUTATIONS:
            logger.warning(
                "ScoreConfig.n_permutations=%d floors every trained-vs-control "
                "p-value at 1/(n_permutations+1)=%.4f before Benjamini-Hochberg "
                "correction across controls raises it further; the conventional "
                "0.05/0.01 thresholds may be unreachable.",
                self.n_permutations,
                1.0 / (self.n_permutations + 1),
            )
        return self


class ScoreStep:
    def __init__(self, config: ScoreConfig) -> None:
        self.config = config

    def run(
        self,
        workspace: EvaluationWorkspace,
        seed: int = 710019,
        predict_result: PredictOutput | None = None,
    ) -> ScoreOutput:
        """Score the trained model's predictions for *workspace*.

        Runs as a plain, single (Rank 0) process. When *predict_result* is
        provided, scoring uses that already-gathered in-memory result and
        avoids reloading prediction/control Parquet files. Otherwise it
        reconstructs the result from the run manifest for standalone or
        restartable scoring.
        """
        logger.info("Running Score step...")

        if predict_result is None:
            predict_result = self._load_predict_output(workspace)

        if predict_result.predictions.is_empty():
            raise ValueError(
                "ScoreStep requires PredictOutput.predictions to be "
                "populated (run the Predict step before scoring)."
            )

        predictions_df = predict_result.predictions
        predictions_col = predictions_df["prediction"].to_list()
        labels_col = predictions_df["label"].to_list()
        probabilities_col = (
            predictions_df["probability"].to_list()
            if "probability" in predictions_df.columns
            else None
        )

        task = get_task(workspace.task_type)
        metric_names = self.config.metrics or task.default_metrics()
        logger.info("Computing metrics: %s", metric_names)
        scores = full_scores(
            task,
            predictions_col,
            labels_col,
            probabilities_col,
            metric_names,
            self.config.metric_aggregation,
        )
        undefined = sorted(name for name, value in scores.items() if np.isnan(value))
        if undefined:
            logger.warning(
                "Metric(s) %s are undefined on this test split (e.g. a class "
                "absent from the labels) and are recorded as NaN.",
                undefined,
            )
        logger.info("Scores: %s", scores)

        config_dict = self.config.model_dump(mode="json")
        log_wandb({f"test/{name}": value for name, value in scores.items()})
        metadata_dict = predict_result.metadata.model_dump(mode="json")
        scores_path = update_run_manifest(
            workspace.run_manifest_path,
            "score",
            {"metadata": metadata_dict, "config": config_dict, "scores": scores},
        )

        result = ScoreOutput(
            scores=scores,
            metadata=predict_result.metadata,
            scores_path=scores_path,
            config=config_dict,
        )

        if self.config.bootstrap_enabled:
            self._run_bootstrap(
                workspace,
                predict_result,
                predictions_col,
                labels_col,
                probabilities_col,
                metric_names,
                seed,
                result,
                metadata_dict,
            )

        long_df = build_long_df(
            result.metadata,
            result.scores,
            result.confidence_intervals,
            result.control_comparisons,
        )
        long_path = workspace.comparison_path / self.config.long_filename
        variant = result.metadata.variant if result.metadata is not None else None
        split = result.metadata.split if result.metadata is not None else None
        combined_long_df = upsert_long_df(long_path, long_df, variant, split)
        result.long_df = combined_long_df
        result.long_path = write_parquet(combined_long_df, long_path)

        return result

    @staticmethod
    def _load_predict_output(workspace: EvaluationWorkspace) -> PredictOutput:
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
        return PredictOutput(
            metadata=EvaluationMetadata.model_validate(metadata_dict),
            predictions=pl.read_parquet(predictions_path),
            predictions_path=predictions_path,
            controls={
                control: pl.read_parquet(path)
                for control, path in control_paths.items()
            },
            control_paths=control_paths,
            requested_controls=list(config_dict.get("controls", [])),
            config=config_dict,
        )

    # -- bootstrap resampling / control comparisons --------------------------

    def _run_bootstrap(
        self,
        workspace: EvaluationWorkspace,
        predict_result: PredictOutput,
        predictions_col: list[Any],
        labels_col: list[Any],
        probabilities_col: list[Any] | None,
        metric_names: list[str],
        seed: int,
        result: ScoreOutput,
        metadata_dict: dict[str, Any] | None,
    ) -> None:
        # Seeded from `seed` so resampling and label permutation are fully
        # deterministic given the same config.
        n_examples = len(predictions_col)
        rng = np.random.default_rng(seed)
        task = get_task(workspace.task_type)

        control_arrays = collect_control_arrays(
            predict_result,
            predictions_col,
            labels_col,
            probabilities_col,
            rng,
            task,
        )

        # Convert predictions/labels/probabilities to 1D NumPy arrays once, up
        # front, rather than re-deriving Python lists (via `[col[j] for j in
        # idx]`) on every one of `n_samples` resamples below: array fancy
        # indexing (`arr[idx]`) is done in C and avoids that per-resample,
        # per-element Python-level list allocation.
        ragged = task.is_ragged
        predictions_arr = as_indexable_array(predictions_col, ragged)
        labels_arr = as_indexable_array(labels_col, ragged)
        probabilities_arr = (
            as_indexable_array(probabilities_col, ragged=False)
            if probabilities_col is not None
            else None
        )
        control_arrays_np = {
            control: (
                as_indexable_array(c_preds, ragged),
                as_indexable_array(c_labels, ragged),
                as_indexable_array(c_probs, ragged=False)
                if c_probs is not None
                else None,
            )
            for control, (c_preds, c_labels, c_probs) in control_arrays.items()
        }

        resample_rows: list[dict[str, float]] = []
        control_resample_rows: dict[str, list[dict[str, float]]] = {
            control: [] for control in control_arrays_np
        }

        n_degenerate: dict[str, int] = {name: 0 for name in metric_names}
        for i in range(self.config.n_samples):
            idx = rng.integers(0, n_examples, size=n_examples)

            trained_scores = full_scores(
                task,
                predictions_arr[idx],
                labels_arr[idx],
                probabilities_arr[idx] if probabilities_arr is not None else None,
                metric_names,
                self.config.metric_aggregation,
            )
            control_scores_by_name = {
                control: full_scores(
                    task,
                    c_preds[idx],
                    c_labels[idx],
                    c_probs[idx] if c_probs is not None else None,
                    metric_names,
                    self.config.metric_aggregation,
                )
                for control, (
                    c_preds,
                    c_labels,
                    c_probs,
                ) in control_arrays_np.items()
            }
            for name in metric_names:
                if np.isnan(trained_scores[name]):
                    n_degenerate[name] += 1

            trained_scores["resample"] = i
            resample_rows.append(trained_scores)
            for control, control_scores in control_scores_by_name.items():
                control_scores["resample"] = i
                control_resample_rows[control].append(control_scores)

        for name, count in n_degenerate.items():
            if count:
                logger.warning(
                    "Metric '%s' was degenerate (e.g. a class missing from "
                    "the resample) in %d/%d bootstrap resamples; those "
                    "resamples are excluded from '%s''s confidence interval "
                    "only, other metrics are unaffected.",
                    name,
                    count,
                    self.config.n_samples,
                    name,
                )

        resamples_df = pl.DataFrame(resample_rows)

        alpha = 1.0 - self.config.confidence_interval
        lower_q, upper_q = alpha / 2, 1.0 - alpha / 2

        confidence_intervals: dict[str, ConfidenceInterval] = {
            name: ci_stats(resamples_df[name].to_numpy(), lower_q, upper_q)
            for name in metric_names
        }
        logger.info("Bootstrap confidence intervals: %s", confidence_intervals)

        output_path = write_parquet(
            resamples_df, workspace.stats_path / self.config.output_filename
        )

        control_comparisons = summarize_controls(
            resample_rows,
            control_resample_rows,
            metric_names,
            lower_q,
            upper_q,
            task,
            predictions_arr,
            labels_arr,
            probabilities_arr,
            control_arrays_np,
            self.config.n_permutations,
            rng,
            self.config.metric_aggregation,
        )

        summary_path = update_run_manifest(
            workspace.run_manifest_path,
            "bootstrap",
            {
                "seed": seed,
                "config": result.config,
                "metadata": metadata_dict,
                "confidence_intervals": confidence_intervals,
            },
        )

        result.confidence_intervals = confidence_intervals
        result.output_path = output_path
        result.resamples = resamples_df
        result.summary_path = summary_path
        result.control_comparisons = control_comparisons or None
