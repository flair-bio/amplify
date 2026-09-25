"""Top-level pretrain config.

Composes per-component config classes into a single validated schema.
Use :func:`modules.core.utils.config_loader.load_and_parse` to build an
instance from YAML files and optional CLI dotlist overrides.
"""

import torch
from pydantic import BaseModel, ConfigDict, Field

from modules.pretrain.src.model.modeling_amplify import AMPLIFYModelConfig
from modules.pretrain.src.model.tokenizer import TokenizerConfig
from modules.pretrain.src.scheduler import SchedulerConfig
from modules.pretrain.src.optimizer import OptimizerConfig
from modules.pretrain.src.dataset.collator import CollatorConfig
from modules.pretrain.src.dataset.dataset import DatasetConfig
from modules.pretrain.src.dataset.dataloader import DataLoaderConfig
from modules.pretrain.src.trainer.trainer import DDPConfig, TrainerConfig, WandbConfig


class PretrainConfig(BaseModel):
    """Composition of all pretrain pipeline configs."""

    model_config = ConfigDict(extra="forbid")
    model: AMPLIFYModelConfig
    tokenizer: TokenizerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    collator: CollatorConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    trainer: TrainerConfig
    ddp: DDPConfig
    wandb: WandbConfig = Field(default_factory=WandbConfig)
