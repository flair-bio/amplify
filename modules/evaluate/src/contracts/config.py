"""Evaluation configuration and cross-stage constraints."""

from __future__ import annotations

import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)
_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES = 1000
_DEFAULT_N_PERMUTATIONS = 1000
BASELINE_NAMES = frozenset({"majority_class", "train_mean"})
DEFAULT_SPLIT_NAME = "default"
CLASSICAL_HEAD_TYPES = frozenset({"sklearn_linear", "knn", "random_forest", "xgboost"})
TORCH_EMBEDDING_HEAD_TYPES = frozenset({"torch_linear"})
BaselineName = Literal["majority_class", "train_mean"]
ControlCondition = Literal[
    "untrained",
    "random_trunk",
    "random_head",
    "random_both",
    "scrambled_labels",
    "scrambled_sequences",
]
HeadType = Literal[
    "bilinear512",
    "attention_l1",
    "categorical_jacobian",
    "mlp",
    "torch_linear",
    "sklearn_linear",
    "knn",
    "random_forest",
    "xgboost",
]
EmbeddingPooling = Literal["mean", "cls"]
EmbeddingDtype = Literal["float16", "float32"]
EmbeddingAutocastDtype = Literal["float16", "bfloat16", "float32"]


class ExternalTestSetConfig(BaseModel):
    """A test-only dataset scored with the model fitted on the main dataset.

    It is never used for training, tuning, or model selection. Its labels must
    follow the main dataset's label contract (column names and class indices).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        description="Filesystem-safe label used for this test set's outputs.",
    )
    dataset_repo_id: str = Field(..., min_length=1)
    split: str = Field("test", min_length=1)


class DatasetSplitConfig(BaseModel):
    """Map dataset-specific split names onto evaluation pipeline roles.

    With only ``name`` set, roles resolve to ``<role>_<name>`` (``train``,
    ``validation``, ``test`` for the ``default`` name). Set a role explicitly
    to point at a split that doesn't follow that convention.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        DEFAULT_SPLIT_NAME,
        min_length=1,
        description="Name recorded in prediction and score metadata.",
    )
    train: str = "train"
    validation: str | None = "validation"
    test: str = "test"
    refit_additional_splits: list[str] = Field(
        default_factory=list,
        description=(
            "Source splits excluded from selection and appended only to the "
            "final train+validation refit."
        ),
    )

    @model_validator(mode="after")
    def _derive_split_names(self) -> DatasetSplitConfig:
        """Suffix each role not set explicitly with a non-default split name."""
        if self.name != DEFAULT_SPLIT_NAME:
            for role in ("train", "validation", "test"):
                if role not in self.model_fields_set:
                    setattr(self, role, f"{role}_{self.name}")
        if any(not split.strip() for split in self.refit_additional_splits):
            raise ValueError("refit_additional_splits cannot contain empty names.")
        if len(self.refit_additional_splits) != len(set(self.refit_additional_splits)):
            raise ValueError("refit_additional_splits must contain unique split names.")
        role_sources = {
            source
            for source in (self.train, self.validation, self.test)
            if source is not None
        }
        overlap = role_sources.intersection(self.refit_additional_splits)
        if overlap:
            raise ValueError(
                "refit_additional_splits must not repeat train, validation, or "
                f"test source splits: {sorted(overlap)}."
            )
        return self


class DatasetWorkspaceConfig(BaseModel):
    """Dataset/task settings shared by model and baseline runs."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str  # dataset name that serves as a mnemonic for the dataset being evaluated, e.g. "stability_prediction"
    dataset_repo_id: str | None = (
        None  # path to HuggingFace dataset repo e.g. biomap-research/stability_prediction
    )
    dataset_revision: str | None = None
    edition: Literal[14, 15, 16] | None = None
    task_type: str
    default_task_type: str | None = None  # dataset default evaluation method
    target_type: Literal["categorical", "continuous"]
    label_format: Literal[
        "scalar", "per_token", "multilabel", "contact_pairs", "consensus_profile"
    ]
    num_labels: int | None = (
        None  # number of classes for *_classification tasks; unused (forced to 1) for sequence_regression
    )
    base_path: str = "./eval"  # base path where the EvaluationWorkspace is stored
    output_dir: Path | None = None

    @field_validator("dataset_name")
    @classmethod
    def _validate_dataset_label(cls, value: str) -> str:
        return validate_directory_label(value)

    @property
    def benchmark_root(self) -> Path:
        return self.output_dir if self.output_dir is not None else Path(self.base_path)

    @property
    def dataset_root(self) -> Path:
        return self.benchmark_root / self.task_type / self.dataset_name

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        from modules.evaluate.src.tasks import available_tasks

        if value not in available_tasks():
            raise ValueError(
                f"Unknown task_type '{value}'. Available: {available_tasks()}."
            )
        return value

    @field_validator("default_task_type")
    @classmethod
    def _validate_default_task_type(cls, value: str | None) -> str | None:
        if value is not None:
            cls._validate_task_type(value)
        return value

    @model_validator(mode="after")
    def _validate_target_metadata(self) -> DatasetWorkspaceConfig:
        if self.target_type == "continuous":
            if self.label_format != "scalar" or self.num_labels is not None:
                raise ValueError(
                    "target_type='continuous' requires label_format='scalar' and "
                    "num_labels=null."
                )
        elif self.label_format in {"scalar", "per_token", "multilabel"} and (
            self.num_labels is None or self.num_labels <= 1
        ):
            raise ValueError("Categorical targets require num_labels > 1.")
        return self


class EvaluationWorkspaceConfig(DatasetWorkspaceConfig):
    """Paths and model settings for one model/dataset evaluation."""

    model_name: str
    run_id: str
    model_repo_id: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tmp_path: str | None = None

    @field_validator("run_id", "model_name")
    @classmethod
    def _validate_directory_label(cls, value: str) -> str:
        return validate_directory_label(value)


def validate_directory_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "Output labels must start with a letter or digit and contain "
            "only letters, digits, '.', '_', or '-'."
        )
    return value


class EmbeddingCacheConfig(BaseModel):
    """Config for the frozen-trunk embedding cache used by :class:`PrepareStep`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    pooling: EmbeddingPooling = Field(
        "mean",
        description=(
            "How to pool (B, S, H) trunk hidden states into (B, H). Must "
            "use 'mean' for residue-only masked mean pooling, or 'cls' "
            "for first-token pooling."
        ),
    )
    layers: list[int] | None = Field(
        default=None,
        description=(
            "Indices into the model's returned hidden-state tuple. Negative "
            "indices follow Python indexing; multiple layers are pooled "
            "independently then concatenated. If omitted while the cache is "
            "enabled, [-1] selects the final layer."
        ),
    )
    dtype: EmbeddingDtype = Field(
        "float32", description="Precision embeddings are stored/computed in."
    )
    autocast_dtype: EmbeddingAutocastDtype = Field(
        "float32",
        description=(
            "Precision used inside the frozen-trunk forward pass. float32 "
            "disables autocast; float16/bfloat16 autocast is opt-in."
        ),
    )
    normalize_before_pooling: bool = Field(
        True,
        description=(
            "L2-normalize each token representation before mean pooling. "
            "Disable explicitly to retain raw representation magnitudes."
        ),
    )
    max_cache_gb: float = Field(
        2.0,
        description=(
            "Refuse to write a split's embeddings to disk if they would "
            "exceed this many GB; falls back to an in-memory-only cache for "
            "that run instead of risking filling shared storage."
        ),
    )
    cache_policy: Literal["reuse", "refresh", "off"] = Field(
        "reuse",
        description=(
            "'reuse': load a matching cached split from disk if present. "
            "'refresh': always recompute and overwrite. 'off': compute "
            "in-memory only, never read or write the disk cache."
        ),
    )

    @field_validator("layers")
    @classmethod
    def _validate_layers(cls, layers: list[int] | None) -> list[int] | None:
        if layers is None:
            return None
        if not layers:
            raise ValueError(
                "embedding_cache.layers must contain at least one layer index."
            )
        if len(set(layers)) != len(layers):
            raise ValueError(
                "embedding_cache.layers must not contain duplicate indices."
            )
        return layers

    @property
    def resolved_layers(self) -> list[int]:
        """Return the effective layers for an enabled embedding cache."""
        return self.layers if self.layers is not None else [-1]


class HeadConfig(BaseModel):
    """Selects and parameterizes the prediction head trained by TuneStep."""

    model_config = ConfigDict(extra="forbid")

    def validate_task(self, task_type: str) -> None:
        from modules.evaluate.src.tasks import get_task

        get_task(task_type).validate_head(self.head_type)

    head_type: HeadType = Field(
        "mlp",
        description=(
            "'torch_linear': one torch.nn.Linear layer trained by gradient "
            "descent. With embedding_cache.enabled=True it is constructed "
            "over cached embeddings; without the cache, the loaded native "
            "task head must itself be a single nn.Linear. 'mlp': existing "
            "native model-head "
            "path. 'bilinear512'/'attention_l1': fixed CASP contact predictors. "
            "'sklearn_linear'/'knn'/'random_forest'/'xgboost': "
            "classical scikit-learn-compatible estimators fit once on cached "
            "pooled embeddings -- require "
            "steps.prepare.embedding_cache.enabled=True in tune_probe mode."
        ),
    )
    hidden_size: int = Field(
        512,
        gt=0,
        description=(
            "Hidden layer width for an MLP probe over cached embeddings; "
            "ignored for other head types."
        ),
    )
    hidden_depth: int = Field(
        1,
        gt=0,
        description=(
            "Number of hidden Linear-GELU layers for a cached MLP probe; "
            "ignored for other head types and native model heads."
        ),
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor kwargs for the chosen estimator (ignored for 'mlp').",
    )
    search_space: dict[str, list[Any]] = Field(
        default_factory=dict,
        description=(
            "Hyperparameter grid swept via sklearn.model_selection.ParameterGrid "
            "(every combination is fit and scored on the validation split; "
            "ignored for 'mlp'). Empty means no sweep: fit once with "
            "'hyperparameters'."
        ),
    )


class PrepareConfig(BaseModel):
    """Config. for the prepare step in evaluation."""

    model_config = ConfigDict(extra="forbid")
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    batch_size: int = 16
    packed: bool = False
    max_tokens_per_batch: int | None = Field(None, gt=0)
    embedding_batch_size: int = Field(
        128,
        gt=0,
        description=(
            "Batch size used only while extracting frozen-trunk embeddings. "
            "Increase until the trunk approaches device memory capacity."
        ),
    )
    dataloader_num_workers: int = Field(
        8,
        ge=0,
        description=(
            "Number of dataloader worker processes per rank and per loader. "
            "Defaults to 8; set to 0 to load batches in the main process."
        ),
    )
    max_length: int = Field(
        512, description="Maximum sequence length after truncation."
    )
    pad_to_multiple_of: int | None = 8
    sequence_column: str = "sequence"
    label_column: str = Field(
        "targets",
        description=(
            "Dataset column containing label(s). Standardized evaluation "
            "datasets use 'targets', not the source field 'label'."
        ),
    )
    id_column: str = Field(
        "id",
        description=(
            "Dataset column containing a per-example identifier, passed "
            "through the batch so downstream prediction/scoring steps can key "
            "predictions/labels back to their source example. Falls back "
            "to a positional index when the dataset has no such column."
        ),
    )
    tokenize_num_proc: int = Field(
        8,
        gt=0,
        description=(
            "Number of processes used to tokenize the dataset up front. "
            "Defaults to 8; must be positive and is independent of dataloader workers."
        ),
    )
    random_truncate_by_split: dict[str, bool] = Field(
        default_factory=lambda: {
            "train": True,
            "validation": False,
            "test": False,
        },
        description=(
            "Whether sequences longer than max_length are cropped from a "
            "random window, keyed by split. Defaults to random train crops "
            "and deterministic validation/test crops."
        ),
    )
    validation_fraction: float = Field(
        0.1,
        description=(
            "Fraction of 'train' held out to synthesize a 'validation' split "
            "when the dataset doesn't already have one."
        ),
    )
    split: DatasetSplitConfig = Field(
        default_factory=DatasetSplitConfig,
        description=(
            "Named dataset split method and source split names to map onto "
            "the pipeline's train/validation/test roles and final-refit-only data."
        ),
    )
    length_grouped_sampling: bool = Field(
        False,
        description=(
            "Batch examples of similar tokenized length together (via "
            "LengthGroupedSampler) instead of pure random shuffling. Reduces "
            "the amount of padding the collator has to add per batch, at the "
            "cost of batches no longer being uniformly randomly composed. "
            "Frozen-trunk extraction groups lengths deterministically; "
            "cached feature batches are not grouped."
        ),
    )
    embedding_cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)


class HyperparameterSearchSpaceConfig(BaseModel):
    """Search space options used by hyperparameter optimization."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: list[float] = [5e-4, 1e-4, 5e-5, 1e-5]
    max_epochs: list[int] = [5, 10, 20]
    weight_decay: list[float] = [0.0, 1e-4, 1e-3]
    weight_decay_sampling: Literal["categorical", "log"] = Field(
        "categorical",
        description=(
            "TPE weight-decay sampling: categorical candidates or continuous "
            "log-scale min/max bounds. Random/grid searches retain the listed "
            "candidates. Log sampling requires finite, positive bounds."
        ),
    )
    batch_size: list[PositiveInt] | None = Field(
        None,
        min_length=1,
        description="Optional training batch sizes for cached MLP probes; null keeps tune.batch_size.",
    )
    hidden_size: list[PositiveInt] | None = Field(
        None,
        min_length=1,
        description="Optional hidden widths for cached MLP probes; null keeps head.hidden_size.",
    )
    hidden_depth: list[PositiveInt] | None = Field(
        None,
        min_length=1,
        description="Optional hidden depths for cached MLP probes; null keeps head.hidden_depth.",
    )

    @model_validator(mode="after")
    def _validate_weight_decay_bounds(self) -> HyperparameterSearchSpaceConfig:
        if self.weight_decay_sampling == "log" and (
            not self.weight_decay
            or any(
                not math.isfinite(value) or value <= 0 for value in self.weight_decay
            )
        ):
            raise ValueError(
                "Log weight_decay sampling requires finite, positive bounds."
            )
        return self


class HyperparameterSearchConfig(BaseModel):
    """Config. for optional hyperparameter search during tuning."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_trials: int = 5
    direction: Literal["maximize", "minimize"] | None = None
    metric: str | None = None
    search_pattern: Literal["bohb", "random", "grid"] = "bohb"
    search_space: HyperparameterSearchSpaceConfig = Field(
        default_factory=HyperparameterSearchSpaceConfig
    )


class TuneConfig(BaseModel):
    """Config for the Tune step: probe training (`tune_probe`) or trunk fine-tuning (`finetune`)."""

    model_config = ConfigDict(extra="forbid")

    head: HeadConfig = Field(default_factory=HeadConfig)

    metric_aggregation: Literal["micro", "macro", "weighted", "samples"] = Field(
        "weighted",
        description=(
            "Averaging used for validation-set model selection (early "
            "stopping, HPO, classical-head sweeps). Auto-synced from "
            "steps.score.metric_aggregation unless set explicitly."
        ),
    )

    # Default (single-run) hyperparameters, used whenever
    # hyperparameter_search.enabled=False.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 10
    warmup_ratio: float = Field(
        0.06, description="Fraction of total training steps used for LR warmup."
    )
    max_grad_norm: float = 1.0
    accumulation_steps: int = Field(
        1,
        gt=0,
        description="Number of batches to accumulate before each optimizer step.",
    )
    batch_size: int | None = Field(
        128,
        gt=0,
        description=(
            "Batch size for cached-embedding probe training and validation. "
            "Set to null to use the batch size created by PrepareStep."
        ),
    )
    standardize_features: bool = Field(
        True,
        description=(
            "Standardize ordinary cached MLP/torch_linear probe features with "
            "training statistics. Layerwise probes use their own setting."
        ),
    )
    early_stopping_patience: int | None = Field(
        5,
        description=(
            "Stop a training run early if the validation metric hasn't "
            "improved for this many epochs, restoring the best epoch's "
            "weights."
        ),
    )

    optimizer: str = Field(
        "adamw",
        description="Optimizer name: adamw, adam, sgd, rmsprop, adagrad, or adafactor.",
    )
    optimizer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra constructor kwargs for the chosen optimizer.",
    )
    lr_scheduler_type: str = Field(
        "linear",
        description="Scheduler name passed to transformers.get_scheduler.",
    )
    scheduler_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra scheduler-specific kwargs.",
    )

    hyperparameter_search: HyperparameterSearchConfig = Field(
        default_factory=HyperparameterSearchConfig
    )
    refit_on_train_validation: bool = Field(
        False,
        description=(
            "After validation selection, initialize a fresh cached probe and fit "
            "on train+validation plus any configured final-refit-only splits "
            "for the selected full epoch budget, without early stopping. "
            "Requires nonempty train/validation embedding datasets; live/native "
            "and layerwise probes are not yet supported."
        ),
    )

    save_best_model: bool = Field(
        True,
        description="Whether to persist the best-performing model and tokenizer.",
    )

    variance_seeds: list[int] = Field(
        default_factory=list,
        description=(
            "Extra seeds to retrain the selected hyperparameters with, purely "
            "to log validation-metric mean/std across training seeds. The "
            "canonical model returned by this step still uses steps.seed; "
            "only cached-embedding probes support this (seeds are cheap there)."
        ),
    )


class PredictConfig(BaseModel):
    """Config. for the predict step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    embedding_batch_size: int | None = Field(
        None,
        gt=0,
        description=(
            "Cached-feature rows per prediction batch, including controls. "
            "Token caches count residues, not proteins. Null inherits the "
            "prepared loader size; live trunk inference is unchanged."
        ),
    )
    collect_probabilities: bool = Field(
        False,
        description=(
            "Whether to collect per-example per-class probabilities "
            "for the trained model and any model-based controls. Required "
            "if ScoreConfig.metrics includes 'roc_auc'. When using the "
            "top-level EvaluationConfig, this is auto-enabled when needed."
        ),
    )
    predictions_filename: str = "predictions.parquet"
    persist_to_disk: bool = Field(
        True,
        description=(
            "Write predictions and the predict manifest to disk. Disable for "
            "fast in-process evaluation when ScoreStep receives this step's "
            "PredictOutput directly."
        ),
    )
    store_sequences: bool = Field(
        False,
        description=(
            "Whether to decode and persist input sequences alongside primary "
            "predictions. Kept off by default to reduce prediction-time "
            "memory use and parquet size. This only affects the main "
            "predictions table; the scrambled_sequences control still stores "
            "its transformed sequences regardless."
        ),
    )

    # Control conditions to compute predictions for, so ScoreStep can compare
    # the trained model against them:
    # - "untrained": the pretrained trunk with a freshly-initialized head,
    #   i.e. the model as it was before fine-tuning.
    # - "random_trunk": independently tuned/refitted probe on random features
    #   for cached probes; a trunk-replacement ablation for live/native runs.
    # - "random_head"/"random_both": the trained model with its head/both
    #   randomly re-initialized, without retraining.
    # - "scrambled_labels": no inference needed -- ScoreStep derives this
    #   directly from the trained model's own predictions/labels.
    # - "scrambled_sequences": the trained model's predictions on inputs
    #   whose token order has been shuffled (sequence-level tasks only).
    controls: list[ControlCondition] = Field(default_factory=lambda: ["random_trunk"])
    baselines: list[BaselineName] = Field(
        default_factory=list,
        description="Inline training-only dataset heuristics, separate from model controls.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_baselines(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        raw_controls = value.get("controls", ["random_trunk"])
        raw_baselines = value.get("baselines", [])
        if not all(
            isinstance(names, (list, tuple))
            and all(isinstance(name, str) for name in names)
            for names in (raw_controls, raw_baselines)
        ):
            return value
        controls = list(raw_controls)
        baselines = list(raw_baselines)
        legacy = [name for name in controls if name in BASELINE_NAMES]
        if legacy:
            warnings.warn(
                "Dataset baselines in predict.controls are deprecated; use "
                "predict.baselines (or reusable baselines.names for sequence tasks).",
                DeprecationWarning,
                stacklevel=2,
            )
            value["controls"] = [name for name in controls if name not in legacy]
            baselines.extend(legacy)
        if len(set(controls)) != len(controls) or len(set(baselines)) != len(baselines):
            raise ValueError(
                "Prediction controls and baselines must not contain duplicates."
            )
        value["baselines"] = baselines
        return value

    @property
    def conditions(self) -> list[str]:
        return [*self.controls, *self.baselines]


class ScoreConfig(BaseModel):
    """Config. for the score step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] | None = None
    metric_aggregation: Literal["micro", "macro", "weighted", "samples"] = "weighted"

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


class BaselinesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[BaselineName] = Field(default_factory=list)

    def validate_task(self, task_type: str) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("baselines.names must not contain duplicates.")
        from modules.evaluate.src.tasks import available_tasks, get_task

        supported = get_task(task_type).dataset_baselines
        for name in self.names:
            if name not in supported:
                providers = [
                    key
                    for key in available_tasks()
                    if name in get_task(key).dataset_baselines
                ]
                raise ValueError(
                    f"Shared baseline '{name}' requires one of task_type={providers}."
                )


class LayerwiseConfig(BaseModel):
    """Configure independent linear probes over frozen hidden-state depths.

    Layers index the trunk's hidden-state tuple: 0 is the token embedding
    output (with mean pooling, an amino-acid composition baseline) and -1 is
    the final block's output after the trunk's final norm; other positions are
    the un-normalized residual stream. Pooling and ``normalize_before_pooling``
    apply to sequence tasks only; token tasks probe unpooled residues. Autocast affects feature
    extraction, not head updates. ``standardize_features`` z-scores each
    layer's features with train-split statistics. Heads train with
    ``steps.tune.learning_rate``; each layer keeps its best validation epoch.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    layers: Literal["all"] | list[int] = "all"
    pooling: Literal["mean", "cls"] = "mean"
    normalize_before_pooling: bool = True
    autocast_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
    standardize_features: bool = True

    @field_validator("layers")
    @classmethod
    def _validate_layers(
        cls, layers: Literal["all"] | list[int]
    ) -> Literal["all"] | list[int]:
        """Check explicit indices; model-dependent bounds and aliases resolve later."""
        if isinstance(layers, list) and (not layers or len(layers) != len(set(layers))):
            raise ValueError("layerwise.layers must contain distinct layer indices.")
        return layers


class StepsConfig(BaseModel):
    """Composition of all pipeline step configs."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tune_probe", "finetune", "evaluate_as_is"] = "tune_probe"
    layerwise: LayerwiseConfig = Field(default_factory=LayerwiseConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)


class WandbConfig(BaseModel):
    """Config. for optional Weights & Biases logging."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "flair-probe"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Overall config. for the evaluation pipeline."""

    model_config = ConfigDict(extra="forbid")
    seed: int = 710019
    tf32: bool = True

    workspace: EvaluationWorkspaceConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    steps: StepsConfig = Field(default_factory=StepsConfig)
    # Test-only datasets scored with the model fitted on the workspace dataset.
    external_test_sets: list[ExternalTestSetConfig] = Field(default_factory=list)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)

    @model_validator(mode="after")
    def _validate_external_test_sets(self) -> EvaluationConfig:
        names = [external.name for external in self.external_test_sets]
        if len(names) != len(set(names)):
            raise ValueError("external_test_sets names must be unique.")
        return self

    @property
    def variant(self) -> str:
        """Filesystem-safe label for this run's evaluation condition."""
        from modules.evaluate.src.tasks import get_task

        mode = self.steps.mode
        if (
            mode == "evaluate_as_is"
            and get_task(self.workspace.task_type).uses_predictor_identity
        ):
            variant = f"{self.steps.tune.head.head_type}-frozen_trunk"
        elif mode == "evaluate_as_is":
            variant = "pretrained"
        elif self.steps.layerwise.enabled:
            variant = "layerwise-torch_linear-frozen_trunk"
        else:
            trunk_suffix = "tuned_trunk" if mode == "finetune" else "frozen_trunk"
            variant = f"{self.steps.tune.head.head_type}-{trunk_suffix}"
        split = self.steps.prepare.split.name
        return variant if split == DEFAULT_SPLIT_NAME else f"{variant}-{split}"

    @model_validator(mode="after")
    def _validate_cross_step_dependencies(self) -> EvaluationConfig:
        self.baselines.validate_task(self.workspace.task_type)
        from modules.evaluate.src.tasks import get_task

        if (
            set(self.steps.predict.baselines)
            & get_task(self.workspace.task_type).dataset_baselines
        ):
            raise ValueError(
                "Move sequence majority_class/train_mean from steps.predict.controls "
                "to baselines.names and run run_evaluate.py --baselines first."
            )
        from modules.evaluate.src.contracts.validation import (
            validate_cross_step_dependencies,
        )

        validate_cross_step_dependencies(self)
        return self


class BaselineRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: DatasetWorkspaceConfig
    prepare: PrepareConfig = Field(
        default_factory=lambda: PrepareConfig.model_validate({})
    )
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    external_test_sets: list[ExternalTestSetConfig] = Field(default_factory=list)
    seed: int = 710019

    @model_validator(mode="after")
    def validate_baselines(self) -> BaselineRunConfig:
        from modules.evaluate.src.tasks import get_task

        self.baselines.validate_task(self.workspace.task_type)
        if not self.baselines.names:
            raise ValueError("Select at least one baseline in baselines.names.")
        names = [external.name for external in self.external_test_sets]
        if len(set(names)) != len(names):
            raise ValueError("external_test_sets names must be unique.")
        task = get_task(self.workspace.task_type)
        task.validate_dataset_config(
            self.workspace.target_type,
            self.workspace.label_format,
            self.workspace.dataset_name,
        )
        return self
