"""Typed data contracts passed between evaluation pipeline steps.

These dataclasses formalize the objects handed off between ``PrepareStep``,
``TuneStep``, ``PredictStep``, and ``ScoreStep`` so that each step has an
explicit, self-documenting contract to implement against instead of an
untyped ``Any``/dict payload described only in comments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.utils.embed import EmbeddingBackend


@dataclass
class PreparedArtifacts:
    """Output of :class:`~modules.evaluate.src.steps.prepare.PrepareStep`.

    Consumed by :class:`~modules.evaluate.src.steps.tune.TuneStep` (which
    returns an updated instance with a fine-tuned ``model``) and by
    :class:`~modules.evaluate.src.steps.predict.PredictStep`.
    """

    model: nn.Module | BaseEstimator
    tokenizer: PreTrainedTokenizerBase
    train_dataloader: DataLoader | None
    val_dataloader: DataLoader | None
    test_dataloader: DataLoader | None

    # Populated only when steps.prepare.embedding_cache.enabled=True, since
    # ``test_dataloader`` then yields pooled ``inputs_embeds`` batches instead
    # of tokens. Every embedding-cache-specific detail PredictStep's control
    # building needs (a re-embeddable trunk, the original tokenized test
    # data, pooling/layers/dtype/hidden size) lives behind this one field --
    # see :class:`~modules.evaluate.src.utils.embed.EmbeddingBackend`.
    # TuneStep/ScoreStep never need to look at it.
    embedding_backend: EmbeddingBackend | None = None

    # Named benchmark condition used to construct train/validation/test.
    split: str = "default"


class EvaluationMetadata(BaseModel):
    """Identifying context for a :class:`PredictOutput`.

    Kept separate from ``predictions`` so downstream consumers (leaderboards,
    aggregation scripts, a future ``ScoreStep``) can tell which
    model/dataset/split a result came from without re-deriving it from the
    workspace. ``split`` identifies the configured benchmark condition (for
    example a sequence-identity threshold or mutation-distance split).

    A Pydantic model rather than a plain dataclass -- unlike the other
    contracts in this module, every field here is plain JSON-safe scalar
    data that crosses a real serialization boundary (``run_manifest.json``),
    so ``.model_dump()``/``.model_validate()`` replace the
    ``dataclass_to_dict``/``EvaluationMetadata(**dict)`` round trip with one
    that's actually validated on reload (see ``ScoreStep._load_predict_output``).
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    dataset_name: str
    task_type: str
    split: str = "default"
    model_repo_id: str | None = None
    dataset_repo_id: str | None = None

    # Which steps.tune.head.head_type produced these predictions (e.g.
    # "mlp"/"sklearn_linear"/"knn"/"random_forest"/"xgboost"/"torch_linear"),
    # so results from different probe types on the same model/dataset can be
    # told apart. None when PredictStep wasn't given one (e.g. direct
    # PredictStep.run() calls that don't go through the full pipeline, or a
    # "pretrained" run where steps.tune never ran and head_type doesn't
    # apply).
    head_type: str | None = None

    # EvaluationConfig.variant: a filesystem-safe label for the full
    # (head_type, trunk-tuning) condition, e.g. "pretrained",
    # "mlp-tuned_trunk", "sklearn_linear-frozen_trunk". Matches the
    # EvaluationWorkspace subdirectory these results were written under, so
    # any saved artifact can be traced back to exactly which condition
    # produced it -- and downstream comparison tables (scores_long.parquet)
    # can group/filter by it directly.
    variant: str | None = None

    # The pipeline seed the producing step ran with. Recorded so any saved
    # artifact/metadata can be traced back to the exact run that produced it
    # (seed-dependent randomness includes control scrambling/reinitialization),
    # without needing to cross-reference a separate run log.
    seed: int | None = None


@dataclass
class PredictOutput:
    """Output of :class:`~modules.evaluate.src.steps.predict.PredictStep`.

    Intended to be consumed by a future metrics/scoring step that computes
    task-specific metrics from the per-example predictions here.
    """

    # Identifies which model/dataset/split produced these predictions.
    metadata: EvaluationMetadata

    # Per-example id/prediction/label (and, if requested, per-class
    # ``probability``) for the model under evaluation, over the test split.
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


@dataclass
class ScoreOutput:
    """Output of :class:`~modules.evaluate.src.steps.score.ScoreStep`.

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
    output_path: Path | None = None

    # Raw per-resample metric values, kept for downstream inspection/plots.
    resamples: pl.DataFrame | None = None

    # Path to the persisted {"metadata", "confidence_intervals"} summary JSON.
    summary_path: Path | None = None

    # Control name -> metric name -> comparison stats. Quantifies how much
    # better the trained model performs than each control condition (e.g. an
    # untrained model, a model with a randomly re-initialized trunk/head, or
    # scrambled labels/sequences), with bootstrap confidence intervals and a
    # permutation-style p-value for the paired difference. Not persisted as
    # its own JSON file: it's only ever written to disk in tidy form, via
    # ``long_df``/``long_path`` below.
    control_comparisons: dict[str, dict[str, ControlComparison]] | None = None

    # Tidy/long-format view of everything above (one row per metric, or per
    # (metric, control) pair), so downstream plotting/aggregation code can
    # ``pl.concat`` the ``long_df``/``long_path`` output of several ScoreStep
    # runs (e.g. one per model) without needing to know how to parse
    # ``run_manifest.json``'s "score"/"bootstrap" sections separately. See
    # ``modules.evaluate.src.utils.long_format.build_long_df``.
    long_df: pl.DataFrame | None = None
    long_path: Path | None = None

    # The ``ScoreConfig`` (as a plain dict, via ``model_dump()``) used to
    # produce ``scores``/``confidence_intervals``/``control_comparisons``, so
    # the settings needed to reproduce this step (metrics, bootstrap sample
    # count, confidence level, etc.) travel with the result.
    config: dict[str, Any] | None = None


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
