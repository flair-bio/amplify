"""Top-level evaluate pipeline config."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from modules.evaluate.src.dataset.dataloader import DEFAULT_SPLIT_NAME
from modules.evaluate.src.dataset.workspace import EvaluationWorkspaceConfig
from modules.evaluate.src.steps.predict import PredictConfig
from modules.evaluate.src.steps.prepare import PrepareConfig
from modules.evaluate.src.steps.score import ScoreConfig
from modules.evaluate.src.steps.tune import TuneConfig
from modules.evaluate.src.utils.config_validation import (
    validate_cross_step_dependencies,
)


class LayerwiseConfig(BaseModel):
    """Configure independent linear probes over frozen hidden-state depths.

    Layers index the trunk's hidden-state tuple: 0 is the token embedding
    output (with mean pooling, an amino-acid composition baseline) and -1 is
    the final block's output after the trunk's final norm; other positions are
    the un-normalized residual stream. Pooling and normalization apply to
    sequence representations, not live token probes. Autocast affects feature
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

    workspace: EvaluationWorkspaceConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    steps: StepsConfig = Field(default_factory=StepsConfig)

    @property
    def variant(self) -> str:
        """Filesystem-safe label for this run's evaluation condition."""
        mode = self.steps.mode
        if mode == "evaluate_as_is":
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
        validate_cross_step_dependencies(self)
        return self
