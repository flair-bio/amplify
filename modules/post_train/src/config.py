"""Strict, fresh-run-only post-training configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., min_length=1)
    revision: str | None = Field(None, min_length=1)
    manifest_file: str = "dataset.json"

    @model_validator(mode="after")
    def validate_source(self) -> DatasetConfig:
        if not self.path.strip():
            raise ValueError(
                "dataset.path must be a local directory or Hub dataset ID."
            )
        path = Path(self.manifest_file)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
            raise ValueError(
                "manifest_file must be a relative JSON path within the dataset."
            )
        return self


class StudentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Path
    # Checkpoint-owned Python must be explicitly trusted; never fetch model code.
    trust_remote_code: bool = False


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = Field("flair-post-train", min_length=1)
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)


class AlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projection_dim: int = Field(512, gt=0)
    pool_proteins: int = Field(256, gt=0)
    loss_weight: float = Field(0.5, ge=0, allow_inf_nan=False)
    initial_temperature: float = Field(0.07, gt=0, allow_inf_nan=False)
    min_inverse_temperature: float = Field(0.01, gt=0, allow_inf_nan=False)


class MLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss_weight: float = Field(1.0, ge=0, allow_inf_nan=False)
    mask_probability: float = Field(0.15, gt=0, le=1, allow_inf_nan=False)
    mask_replace_probability: float = Field(0.8, ge=0, le=1, allow_inf_nan=False)
    random_replace_probability: float = Field(0.1, ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_replacements(self) -> MLMConfig:
        if self.mask_replace_probability + self.random_replace_probability > 1:
            raise ValueError(
                "MLM MASK and random replacement probabilities must sum to at most 1."
            )
        return self


AdamBeta = Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    batch_size: int = Field(256, gt=0)
    microbatch_size: int = Field(16, gt=0)
    encoder_lr: float = Field(3e-6, gt=0, allow_inf_nan=False)
    head_lr: float = Field(1e-3, gt=0, allow_inf_nan=False)
    betas: tuple[AdamBeta, AdamBeta] = (0.9, 0.95)
    weight_decay: float = Field(0.01, ge=0, allow_inf_nan=False)
    gradient_clip: float = Field(1.0, gt=0, allow_inf_nan=False)
    validation_interval: int = Field(25, gt=0)
    anchor_chunk_size: int = Field(256, gt=0)

    @model_validator(mode="after")
    def validate_device(self) -> TrainingConfig:
        if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("Use auto, one CUDA device, or CPU for bounded checks.")
        return self


class PostTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["latent_alignment"] = "latent_alignment"
    seed: int = Field(42, ge=0, lt=2**32)
    student: StudentConfig
    dataset: DatasetConfig
    output_dir: Path
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    mlm: MLMConfig = Field(default_factory=MLMConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @model_validator(mode="after")
    def validate_batches(self) -> PostTrainingConfig:
        pool = self.alignment.pool_proteins
        if self.training.batch_size % pool or pool % self.training.microbatch_size:
            raise ValueError(
                "training.batch_size must be a multiple of alignment.pool_proteins; "
                "training.microbatch_size must divide alignment.pool_proteins."
            )
        return self


StructuralAlignmentConfig = PostTrainingConfig
