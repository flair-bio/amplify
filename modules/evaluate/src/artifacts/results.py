"""In-memory evaluation outputs and serialized metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator
from torch import nn

if TYPE_CHECKING:
    from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace


@dataclass
class FitResult:
    """Fitted model and selection provenance, separate from its input data."""

    model: nn.Module | BaseEstimator
    seed: int
    selected_hyperparameters: dict[str, Any] = field(default_factory=dict)
    layer_selections: dict[int, dict[str, float]] = field(default_factory=dict)
    seed_variance: dict[str, list[float]] = field(default_factory=dict)
    final_fit_split: str = "train"
    selection_metric: str | None = None
    selection_score: float | None = None
    input_identity: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, dict[str, Any]] = field(default_factory=dict)
    checkpoint_settings: dict[str, Any] | None = None
    serialization: Literal["distributed", "local"] = "distributed"


class EvaluationMetadata(BaseModel):
    """Identifying context for a :class:`PredictOutput`.

    Kept separate from ``predictions`` so downstream consumers (leaderboards,
    aggregation scripts and scoring functions) can tell which
    model/dataset/split a result came from without re-deriving it from the
    workspace. ``split`` identifies the configured benchmark condition (for
    example a sequence-identity threshold or mutation-distance split).

    A Pydantic model rather than a plain dataclass because its JSON-safe fields
    cross a real serialization boundary in ``run_manifest.json`` and score
    reports.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    dataset_name: str
    task_type: str
    split: str = "default"
    test_subset: str | None = None
    model_repo_id: str | None = None
    dataset_repo_id: str | None = None
    dataset_config_name: str | None = None
    dataset_revision: str | None = None
    group_column: str | None = None

    # Which steps.tune.head.head_type produced these predictions (e.g.
    # "mlp"/"sklearn_linear"/"knn"/"random_forest"/"xgboost"/"torch_linear"),
    # so results from different probe types on the same model/dataset can be
    # told apart. None for direct prediction or an evaluate_as_is run.
    head_type: str | None = None

    # Public hidden-state tuple index for layerwise probe runs. ``None`` for
    # ordinary single-head evaluations.
    layer_index: int | None = None

    # EvaluationConfig.variant: a filesystem-safe label for the full
    # (head_type, trunk-tuning) condition, e.g. "pretrained",
    # "mlp-tuned_trunk", "sklearn_linear-frozen_trunk". Matches the
    # EvaluationWorkspace subdirectory these results were written under, so
    # any saved artifact can be traced back to exactly which condition
    # produced it -- and downstream comparison tables (scores.parquet)
    # can group/filter by it directly.
    variant: str | None = None

    # The pipeline seed the producing step ran with. Recorded so any saved
    # artifact/metadata can be traced back to the exact run that produced it
    # (seed-dependent randomness includes control scrambling/reinitialization),
    # without needing to cross-reference a separate run log.
    seed: int | None = None


class ContactEvaluationMetadata(EvaluationMetadata):
    """Dataset/protocol identity specific to the HF-only CASP scoring contract."""

    dataset_fingerprint: str
    schema_version: str
    protocol_version: str


@dataclass
class PredictOutput:
    """Primary and control predictions consumed by pure scientific scoring."""

    # Identifies which model/dataset/split produced these predictions.
    metadata: EvaluationMetadata

    # Per-example id/prediction/label (and, if requested, per-class
    # ``probability``) for the model under evaluation, over the test split.
    # Layerwise outputs also include ``layer_index`` as a column for easy
    # concatenation across per-layer workspaces.
    predictions: pl.DataFrame
    predictions_path: Path | None = None

    # Control name -> per-example id/prediction (and ``probability``, if
    # collected) dataframe, row-aligned to ``predictions`` by ``id``. Only
    # populated for controls that require model inference (i.e. excludes
    # "scrambled_labels", which needs no inference and is instead derived by
    # downstream scoring directly from ``predictions``' labels).
    controls: dict[str, pl.DataFrame] = field(default_factory=dict)
    control_paths: dict[str, Path] = field(default_factory=dict)

    # Every control requested via ``PredictConfig.controls``, including
    # "scrambled_labels" (which has no entry in ``controls``/``control_paths``
    # above), so downstream scoring knows which comparisons to compute.
    requested_controls: list[str] = field(default_factory=list)

    # The ``PredictConfig`` (as a plain dict, via ``model_dump()``) used to
    # produce ``predictions``/``controls``, so the settings needed to
    # reproduce this step (e.g. which controls were requested) travel with
    # the result even if it's inspected independently of the config that
    # created it.
    config: dict[str, Any] | None = None
    baseline_context: dict[str, Any] | None = None


@dataclass
class ScoreOutput:
    """Flat-task scores and optional statistical comparisons.

    Combines the point-estimate task metrics with (optionally) bootstrap
    confidence intervals and trained-vs-control comparisons.
    """

    # Aggregate metric name -> value, e.g. {"accuracy": 0.91, "f1": 0.88}.
    scores: dict[str, float]

    # Identifies which model/dataset/split produced ``scores``.
    metadata: EvaluationMetadata | None = None

    # Path where ``scores`` was persisted on disk.
    scores_path: Path | None = None

    # Metric name -> bootstrap confidence interval, only populated when
    # ``ScoreConfig.bootstrap_enabled=True``.
    confidence_intervals: dict[str, ConfidenceInterval] | None = None
    # Kept for API compatibility; per-resample parquet output is not persisted.
    output_path: Path | None = None

    # Raw per-resample values retained in memory while computing summaries.
    resamples: pl.DataFrame | None = None

    # Path to the persisted {"metadata", "confidence_intervals"} summary JSON.
    summary_path: Path | None = None

    # Control name -> metric name -> comparison stats. Quantifies how much
    # better the trained model performs than each control condition (e.g. an
    # untrained model, a model with a randomly re-initialized trunk/head, or
    # scrambled labels/sequences), with bootstrap confidence intervals and a
    # permutation-style p-value for the paired difference. Persisted in the
    # run manifest and tidy ``long_df``/``long_path`` output.
    control_comparisons: dict[str, dict[str, ControlComparison]] | None = None

    # Tidy/long-format view of everything above (one row per metric, or per
    # (metric, control) pair), so downstream plotting/aggregation code can
    # ``pl.concat`` the ``long_df``/``long_path`` output of several scoring
    # runs (e.g. one per model) without needing to know how to parse
    # ``run_manifest.json``'s "score"/"bootstrap" sections separately. See
    # ``modules.evaluate.src.metrics.tables.build_long_df``.
    long_df: pl.DataFrame | None = None
    long_path: Path | None = None

    # The ``ScoreConfig`` (as a plain dict, via ``model_dump()``) used to
    # produce ``scores``/``confidence_intervals``/``control_comparisons``, so
    # the settings needed to reproduce this step (metrics, bootstrap sample
    # count, confidence level, etc.) travel with the result.
    config: dict[str, Any] | None = None

    condition_scores: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class StructuredScoreOutput:
    """Unit and aggregate scientific tables, with publication metadata."""

    metadata: EvaluationMetadata
    unit_scores: pl.DataFrame
    summaries: pl.DataFrame
    unit_scores_path: Path | None = None
    summaries_path: Path | None = None
    control_unit_scores: dict[str, pl.DataFrame] = field(default_factory=dict)
    control_summaries: dict[str, pl.DataFrame] = field(default_factory=dict)
    control_unit_scores_paths: dict[str, Path] = field(default_factory=dict)
    control_summaries_paths: dict[str, Path] = field(default_factory=dict)


@dataclass
class ContactScoreOutput(StructuredScoreOutput):
    """CASP unit and hierarchical scores with the established disk contract."""


@dataclass
class EvaluationResult:
    """One prediction/scoring handoff, identified by prediction metadata."""

    workspace: EvaluationWorkspace
    prediction: PredictOutput
    score: ScoreOutput | StructuredScoreOutput | None = None


@dataclass
class PipelineResult:
    """Result records; scores are populated on the main process only."""

    evaluations: list[EvaluationResult]
    # Results of the same fitted model on each configured external test set.
    external: dict[str, PipelineResult] = field(default_factory=dict)
    test_subsets: dict[str, PipelineResult] = field(default_factory=dict)

    @property
    def predictions(self) -> dict[int | None, PredictOutput]:
        """Compatibility view for callers using the original layer-keyed API."""
        return {
            item.prediction.metadata.layer_index: item.prediction
            for item in self.evaluations
        }

    @property
    def scores(self) -> dict[int | None, ScoreOutput | StructuredScoreOutput]:
        return {
            item.prediction.metadata.layer_index: item.score
            for item in self.evaluations
            if item.score is not None
        }


class ConfidenceInterval(TypedDict):
    """A single metric's bootstrap point estimate and confidence interval."""

    mean: float
    lower: float
    upper: float


class ControlComparison(TypedDict):
    """Paired trained-vs-control bootstrap comparison for a single metric.

    ``diff_mean``/``diff_lower``/``diff_upper`` describe the bootstrap
    distribution of ``trained_score - control_score`` per resample;
    ``p_value`` is the paired permutation-test fraction where the trained
    model fails to beat the control (i.e. ``diff <= 0``). ``p_value_adjusted``
    is the Benjamini-Hochberg-corrected FDR q-value, computed across controls
    within each metric produced by a single bootstrap run, to guard against
    false positives from the resulting multiple comparisons.
    """

    trained_mean: float
    control_mean: float
    diff_mean: float
    diff_lower: float
    diff_upper: float
    p_value: float
    p_value_adjusted: float
