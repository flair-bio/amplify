"""Runner-owned application setup shared by ordinary and assay evaluations."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from accelerate import Accelerator, PartialState

from modules.evaluate.src.config import BaselineRunConfig, EvaluationConfig
from modules.evaluate.src.utils.config_utils import RunConfig
from modules.evaluate.src.utils.seed import seed_everything


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    accelerator: Accelerator | None = None

    @property
    def is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for name in ("urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def configure_precision(tf32: bool) -> None:
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


@contextmanager
def initialize_runtime(config: RunConfig) -> Iterator[Runtime]:
    """Create one execution context and always release the tracking run."""
    configure_logging()
    workflow = config.workflow
    baseline_only = isinstance(config, BaselineRunConfig)
    if workflow != "evaluation" and PartialState().num_processes != 1:
        raise ValueError("ProteinGym runners require --num_processes 1")
    accelerator = (
        Accelerator(gradient_accumulation_steps=config.steps.tune.accumulation_steps)
        if isinstance(config, EvaluationConfig)
        else None
    )
    device = (
        accelerator.device
        if accelerator is not None
        else torch.device(
            getattr(config, "device", None)
            or ("cpu" if baseline_only or not torch.cuda.is_available() else "cuda")
        )
    )
    runtime = Runtime(device, accelerator)
    if not isinstance(config, BaselineRunConfig):
        configure_precision(config.tf32)
    tracking = None
    try:
        if (
            not isinstance(config, BaselineRunConfig)
            and config.wandb.enabled
            and runtime.is_main_process
        ):
            import wandb

            wandb_config = config.wandb
            workspace = config.workspace
            default_name = (
                f"{workspace.model_name}-{workspace.dataset_name}"
                if workspace is not None
                else workflow
            )
            if isinstance(config, EvaluationConfig):
                default_name = f"{default_name}-{config.variant}"
            tracking = wandb.init(
                project=wandb_config.project,
                entity=wandb_config.entity,
                name=wandb_config.run_name or default_name,
                tags=wandb_config.tags,
                config=config.model_dump(mode="json"),
            )
        seed_everything(config.seed)
        yield runtime
    finally:
        if tracking is not None:
            tracking.finish()
