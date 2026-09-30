import logging
import os
import re
import sys
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import (
    DataLoaderConfiguration,
    InitProcessGroupKwargs,
    TorchDynamoPlugin,
    set_seed,
)
from torch.utils.data import ConcatDataset

from modules.core.utils.config_loader import load_and_parse
from modules.pretrain.src.config import PretrainConfig
from modules.pretrain.src.dataset.collator import get_collator
from modules.pretrain.src.dataset.dataloader import build_split_dataloader
from modules.pretrain.src.dataset.dataset import prepare_sources
from modules.pretrain.src.metric.logger import TrainingLogger
from modules.pretrain.src.model.modeling_amplify import get_amplify_masked_lm
from modules.pretrain.src.model.tokenizer import get_tokenizer
from modules.pretrain.src.optimizer import get_optimizer
from modules.pretrain.src.scheduler import get_scheduler
from modules.pretrain.src.trainer.trainer import Trainer

GLOBAL_RANK = os.environ.get("RANK", "0")

logging.basicConfig(
    level=os.environ.get("PLM_LOG_LEVEL", "INFO"),
    format=f"%(asctime)s,%(msecs)03d | r{GLOBAL_RANK} | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Reduce verbosity of non-main ranks.
if GLOBAL_RANK != "0":
    # Global logger, not scoped to the project. Consider introducing `plm.` project log namespace.
    _log = logging.getLogger()
    _other_ranks_level = logging.WARNING
    _log.debug(
        f"Setting rank_{GLOBAL_RANK} log level: {logging.getLevelName(_other_ranks_level)}"
    )
    _log.setLevel(_other_ranks_level)


def setup_runtime(config: PretrainConfig) -> Accelerator:
    """Initializes Accelerate and hardware/RNG state shared by every run.

    Args:
        config: Loaded pretrain config.

    Returns:
        The constructed ``Accelerator``.
    """
    if config.trainer.expandable_segments:
        # Must be set before the CUDA init that occurs in Accelerator()
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # `drop_last=True` on `config.dataloader` already ensures even batches per rank.
    dataloader_config = DataLoaderConfiguration(even_batches=False, non_blocking=True)
    # Accelerate applies compile after the DDP wrap (PyTorch's recommended
    # order, so DDPOptimizer can overlap gradient all-reduce with backward).
    dynamo_plugin = (
        TorchDynamoPlugin(
            backend="inductor",
            mode=config.trainer.compile_mode,
            fullgraph=config.trainer.compile_fullgraph,
            dynamic=config.trainer.compile_dynamic,
        )
        if config.trainer.compile
        else None
    )
    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        dynamo_plugin=dynamo_plugin,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        log_with="wandb" if config.wandb.enabled else None,
        step_scheduler_with_optimizer=False,  # Prevent Accelerate from over-decaying the LR on multi-GPU; Trainer steps the scheduler manually.
        kwargs_handlers=[
            InitProcessGroupKwargs(
                timeout=timedelta(minutes=config.trainer.dist_timeout_minutes)
            ),
            config.ddp.to_accelerate(),
        ],
    )
    if config.trainer.compile:
        logger.info(
            f"torch.compile enabled (mode={config.trainer.compile_mode}, "
            f"dynamic={config.trainer.compile_dynamic}). First steps will be slow "
            "while graphs are compiled."
        )

    # Seed all RNGs (python/numpy/torch) with dataset.seed for determinism across runs and ranks.
    if config.trainer.deterministic:
        # 8 x 4MiB fixed cuBLAS workspaces: pins kernel choice so matmuls are
        # reproducible. Must precede cuBLAS init or deterministic GEMMs raise.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if accelerator.is_main_process:
            logger.warning(
                "trainer.deterministic=true: slows training and sets "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8. Runs are still not bitwise "
                "reproducible while packing is on because variable-length "
                "attention may use nondeterministic kernels. "
                "For debugging, not production."
            )
    set_seed(config.dataset.seed, deterministic=config.trainer.deterministic)

    # TensorFloat-32 speeds up matmuls (~3x) on Ampere+/Hopper GPUs.
    torch.set_float32_matmul_precision("high" if config.trainer.tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = config.trainer.tf32
    torch.backends.cudnn.allow_tf32 = config.trainer.tf32

    return accelerator


def setup_tracking(config: PretrainConfig, accelerator: Accelerator) -> None:
    """Initializes W&B experiment tracking via Accelerate, if enabled.

    Args:
        config: Loaded pretrain config.
        accelerator: Accelerator to register trackers on.
    """
    if not config.wandb.enabled:
        return

    wandb_config = config.wandb
    # Consolidates logs and checkpoints under one folder by default.
    wandb_dir = wandb_config.dir or (config.trainer.output_dir / "wandb")
    # W&B only resumes via a stable `id`, not by matching `name`; default it
    # to a slug of `run_name` so relaunching (e.g. after preemption) reattaches
    # to the same W&B run automatically.
    wandb_id = wandb_config.id
    if wandb_id is None and wandb_config.run_name:
        wandb_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", wandb_config.run_name).strip("-")

    if accelerator.is_main_process:
        wandb_dir.mkdir(parents=True, exist_ok=True)

    accelerator.init_trackers(
        project_name=wandb_config.project,
        config=config.model_dump(mode="json"),
        init_kwargs={
            "wandb": {
                "entity": wandb_config.entity,
                "name": wandb_config.run_name,
                "tags": wandb_config.tags,
                "mode": wandb_config.mode,
                "dir": str(wandb_dir),
                "id": wandb_id,
                "resume": wandb_config.resume,
            }
        },
    )
    logger.info("W&B tracking initialized.")


def main() -> None:
    """Entry point for the pretrain pipeline.

    Usage::

        uv run run_pretrain.py <config.yaml> [<override.yaml> ...] [key=value ...]

    One or more YAML files may be supplied; later files override earlier ones
    for shared keys.  Any ``key=value`` arguments (containing ``=``) are
    treated as OmegaConf dotlist overrides applied after YAML merging.

    Examples::

        uv run modules/pretrain/src/run_pretrain.py modules/pretrain/configs/config.yaml
        uv run modules/pretrain/src/run_pretrain.py modules/pretrain/configs/config.yaml configs/experiment.yaml
        uv run modules/pretrain/src/run_pretrain.py modules/pretrain/configs/config.yaml model.hidden_size=512
    """
    args = sys.argv[1:]
    config_path = [a for a in args if "=" not in a]
    overrides = [a for a in args if "=" in a] or None

    if not config_path:
        logger.error(
            "Usage: run_pretrain <config.yaml> [<override.yaml> ...] [key=value ...]"
        )
        sys.exit(1)

    logger.info("=== Starting Pretrain Run ===")
    config = load_and_parse(
        path=config_path, model_cls=PretrainConfig, overrides=overrides
    )
    logger.info(f"Config loaded from {config_path}.")

    accelerator = setup_runtime(config)

    if config.dataset.prepare_only:
        # Before the model/tracker exist: a cache build needs neither, so this
        # can run on CPU-only hardware without allocating a GPU.
        with accelerator.local_main_process_first():
            prepare_sources(config.dataset)
        logger.info("Data sources prepared. Exiting (dataset.prepare_only).")
        return

    setup_tracking(config, accelerator)

    model = get_amplify_masked_lm(config.model)
    tokenizer = get_tokenizer(config.tokenizer)
    optimizer = get_optimizer(model=model, config=config.optimizer)
    scheduler = get_scheduler(optimizer=optimizer, scheduler_config=config.scheduler)
    collator = get_collator(tokenizer=tokenizer, collator_config=config.collator)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    logger.info("Model, tokenizer, optimizer, scheduler, and collator instantiated.")

    # Phase 1: filter/split/index each source once, cached to disk (local rank 0 builds first).
    with accelerator.local_main_process_first():
        prepared_sources = prepare_sources(config.dataset)
    logger.info("Data sources prepared.")

    # Built once and reused every epoch — only the per-epoch row *selection*
    # (EpochPlan) changes, not the underlying data these wrap.
    train_concat = ConcatDataset([p.train_dataset for p in prepared_sources])
    val_concat = ConcatDataset([p.val_dataset for p in prepared_sources])

    # Val set has a fixed seed and doesn't change per epoch, so build it once.
    val_dataloader = build_split_dataloader(
        prepared_sources,
        val_concat,
        dataset_config=config.dataset,
        collator=collator,
        dataloader_config=config.dataloader,
        split="val",
        num_processes=accelerator.num_processes,
        process_index=accelerator.process_index,
    )
    val_dataloader = accelerator.prepare(val_dataloader)
    logger.info("Validation dataloader instantiated.")

    # Phase 2: rebuild each epoch's mixture (cluster-dedup + per-source subsampling).
    def train_dataloader_provider(epoch: int) -> torch.utils.data.DataLoader:
        train_dataloader = build_split_dataloader(
            prepared_sources,
            train_concat,
            dataset_config=config.dataset,
            collator=collator,
            dataloader_config=config.dataloader,
            split="train",
            epoch=epoch,
            num_processes=accelerator.num_processes,
            process_index=accelerator.process_index,
        )
        return accelerator.prepare(train_dataloader)

    metric_logger = TrainingLogger()
    trainer = Trainer(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader_provider=train_dataloader_provider,
        config=config.trainer,
        val_dataloader=val_dataloader,
        metric_logger=metric_logger,
        tokenizer=tokenizer,
    )
    logger.info("Trainer instantiated. Starting training loop.")

    trainer.train()


if __name__ == "__main__":
    main()
