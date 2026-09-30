"""Top-level evaluate pipeline config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, model_validator

from modules.evaluate.src.dataset.dataloader import DEFAULT_SPLIT_NAME
from modules.evaluate.src.dataset.workspace import EvaluationWorkspaceConfig
from modules.evaluate.src.steps.prepare import PrepareConfig
from modules.evaluate.src.steps.tune import TuneConfig
from modules.evaluate.src.steps.predict import PredictConfig
from modules.evaluate.src.steps.score import ScoreConfig
from modules.evaluate.src.utils.config_validation import (
    validate_cross_step_dependencies,
)


class StepsConfig(BaseModel):
    """Composition of all pipeline step configs."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tune_probe", "finetune", "evaluate_as_is"] = "tune_probe"
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
        else:
            trunk_suffix = "tuned_trunk" if mode == "finetune" else "frozen_trunk"
            variant = f"{self.steps.tune.head.head_type}-{trunk_suffix}"
        split = self.steps.prepare.split.name
        return variant if split == DEFAULT_SPLIT_NAME else f"{variant}-{split}"

    @model_validator(mode="after")
    def _validate_cross_step_dependencies(self) -> EvaluationConfig:
        validate_cross_step_dependencies(self)
        return self
