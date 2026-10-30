"""Top-level evaluate pipeline config."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.config.score import ScoreConfig
from modules.evaluate.src.dataset.dataloader import (
    DEFAULT_SPLIT_NAME,
    ExternalTestSetConfig,
)
from modules.evaluate.src.dataset.preparation import PrepareConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspaceConfig
from modules.evaluate.src.training.config import TuneConfig
from modules.evaluate.src.utils.baselines import BaselinesConfig
from modules.evaluate.src.utils.config_validation import (
    validate_cross_step_dependencies,
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
    # Canonical ``runtime.data_root``: local mirror for configured datasets.
    # Applied at plan time by run_evaluate.py; recorded here for validation.
    data_root: Path | None = None

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
        mode = self.steps.mode
        if (
            mode == "evaluate_as_is"
            and self.workspace.task_type == "contact_prediction"
            and self.steps.tune.head.head_type == "categorical_jacobian"
        ):
            variant = "categorical_jacobian-frozen_trunk"
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
        if self.workspace.task_type in {
            "sequence_classification",
            "sequence_regression",
        } and set(self.steps.predict.controls) & {"majority_class", "train_mean"}:
            raise ValueError(
                "Move sequence majority_class/train_mean from steps.predict.controls "
                "to baselines.names and run run_evaluate.py --baselines first."
            )
        validate_cross_step_dependencies(self)
        return self
