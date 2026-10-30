"""Typed configuration for probe training and hyperparameter search.

Owned by the training package (not the legacy ``steps`` package) so the fitting
code can be imported without pulling in the legacy pipeline.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from modules.evaluate.src.model.heads import HeadConfig
from modules.evaluate.src.utils.hpo import OPTIMIZER_REGISTRY

logger = logging.getLogger(__name__)


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
    search_pattern: Literal["tpe", "bohb", "random", "grid"] = "tpe"
    search_space: HyperparameterSearchSpaceConfig = Field(
        default_factory=HyperparameterSearchSpaceConfig
    )

    @field_validator("search_pattern", mode="after")
    @classmethod
    def _normalize_search_pattern(cls, value: str) -> str:
        """Store the accurate ``tpe`` name; accept legacy ``bohb`` with a warning."""
        if value == "bohb":
            logger.warning(
                "hyperparameter_search.search_pattern='bohb' is deprecated; use "
                "'tpe' (Optuna TPE + median pruning). Treating it as 'tpe'."
            )
            return "tpe"
        return value

    @property
    def engine_pattern(self) -> Literal["bohb", "random", "grid"]:
        """Pattern name understood by the shared head-search engine (``bohb``)."""
        if self.search_pattern == "tpe":
            return "bohb"
        return self.search_pattern


class TuneConfig(BaseModel):
    """Config for probe training (`tune_probe`) or trunk fine-tuning (`finetune`)."""

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
        description=f"Optimizer name, one of: {sorted(OPTIMIZER_REGISTRY)}.",
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
