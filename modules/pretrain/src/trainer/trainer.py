import logging
import math
import re
import shutil
import signal
import sys
import time
from collections.abc import Callable
from itertools import count
from pathlib import Path
from types import FrameType
from typing import Any, Literal

import torch
from accelerate import Accelerator
from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from modules.pretrain.src.metric.logger import TrainingLogger
from modules.pretrain.src.model.modeling_amplify import register_amplify_auto_classes

logger = logging.getLogger(__name__)


class DDPConfig(BaseModel):
    """DistributedDataParallel tuning parameters. Ignored for single node."""

    model_config = ConfigDict(extra="forbid")

    broadcast_buffers: bool = Field(
        True,
        description=(
            "Enables syncing (broadcasting) buffers of the module at beginning "
            "of the forward pass."
        ),
    )
    bucket_cap_mb: int | None = Field(
        None,
        gt=0,
        description=(
            "DDP will bucket parameters into multiple buckets so that gradient "
            "reduction of each bucket can potentially overlap with backward "
            "computation. bucket_cap_mb controls the bucket size in MiB."
        ),
    )
    gradient_as_bucket_view: bool = Field(
        False,
        description=(
            "Point `.grad` at the reducer's bucket instead of copying out of "
            "it. PyTorch's own default is false. Saves a gradient-sized copy "
            "per bucket and a gradient-sized allocation."
        ),
    )
    static_graph: bool = Field(
        False,
        description="When set to True, DDP knows the trained graph is static",
    )
    find_unused_parameters: bool = Field(False)
    comm_hook: Literal["none", "bf16", "fp16"] = Field(
        "none",
        description="Reduce precision of the all-reduce collective.",
    )

    def to_accelerate(self) -> DistributedDataParallelKwargs:
        """Translates this config into Accelerate's DDP kwargs handler."""
        comm_hooks = {
            "none": DDPCommunicationHookType.NO,
            "bf16": DDPCommunicationHookType.BF16,
            "fp16": DDPCommunicationHookType.FP16,
        }
        return DistributedDataParallelKwargs(
            broadcast_buffers=self.broadcast_buffers,
            bucket_cap_mb=self.bucket_cap_mb,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
            static_graph=self.static_graph,
            comm_hook=comm_hooks[self.comm_hook],
        )


class WandbConfig(BaseModel):
    """Config for optional Weights & Biases experiment tracking."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        False, description="Enable W&B logging via accelerate trackers."
    )
    project: str = Field("flair-pretrain", description="W&B project name.")
    entity: str | None = Field(None, description="W&B entity (team/user).")
    run_name: str | None = Field(None, description="Display name for this run.")
    tags: list[str] = Field(
        default_factory=list, description="Tags attached to the run."
    )
    mode: str = Field(
        "offline", description="W&B mode: 'online', 'offline', or 'disabled'."
    )
    dir: Path | None = Field(
        None,
        description=(
            "Local directory for W&B run data; defaults to "
            "`<trainer.output_dir>/wandb` so logs and checkpoints share one "
            "folder."
        ),
    )
    id: str | None = Field(
        None,
        description=(
            "Explicit W&B run id, for deterministic resume. Defaults to a "
            "slug of `run_name`, so relaunching with the same `run_name` "
            "(e.g. via `resume_from_checkpoint`) resumes the same W&B run."
        ),
    )
    resume: str = Field(
        "allow",
        description=(
            "Resume mode passed to `wandb.init` with `id` ('allow', 'must', "
            "'never', ...); 'allow' creates the run if new, resumes it if not."
        ),
    )


class TrainerConfig(BaseModel):
    """Config for :class:`Trainer`."""

    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(1, gt=0, description="Number of epochs to train for.")
    max_steps: int | None = Field(
        None,
        gt=0,
        description=(
            "If set, overrides `num_epochs`: training continues across as "
            "many epochs as needed until this many optimizer steps are "
            "reached (mirrors HuggingFace Trainer semantics)."
        ),
    )
    max_grad_norm: float = Field(1.0, gt=0.0, description="Gradient clipping norm.")
    gradient_accumulation_steps: int = Field(
        1,
        ge=1,
        description=(
            "Number of micro-batches to accumulate gradients over before an "
            "optimizer step. The effective batch size is this value times the "
            "per-device batch size times the number of processes. Must match "
            "the value the `Accelerator` was constructed with; the Trainer "
            "validates this on init."
        ),
    )
    output_dir: Path = Field(
        Path("./checkpoints"), description="Directory for training-state checkpoints."
    )
    resume_from_checkpoint: Path | None = Field(
        None, description="Path to a checkpoint directory to resume from."
    )
    logging_steps: int = Field(
        50, gt=0, description="Log accumulated metrics every N optimizer steps."
    )
    eval_steps: int | None = Field(None, description="Run evaluation every N steps.")
    save_steps: int | None = Field(
        None, description="Save Accelerate state every N steps."
    )
    hf_save_steps: int | None = Field(
        None, description="Save HF checkpoint every N steps."
    )
    save_total_limit: int | None = Field(
        None,
        gt=0,
        description=(
            "Max resumable (Accelerate) checkpoints kept under `output_dir`; "
            "oldest (by step) deleted once exceeded. Unlimited if unset."
        ),
    )
    hf_save_total_limit: int | None = Field(
        None,
        gt=0,
        description=(
            "Max HF-format checkpoints kept under "
            "`output_dir/hf_checkpoints`; oldest (by step) deleted once "
            "exceeded. Unlimited if unset."
        ),
    )
    tf32: bool = Field(
        True,
        description=(
            "Enable TensorFloat-32 matmuls (~3x speedup on Ampere+/Hopper "
            "GPUs), trading a small amount of numerical precision for speed."
        ),
    )
    expandable_segments: bool = Field(
        True,
        description=(
            "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to minimize "
            "GPU memory allocator fragmentation. This keeps Peak reserved "
            r"memory within ~2% of allocated memory across ranks."
        ),
    )
    deterministic: bool = Field(
        False,
        description=(
            "Enable `torch.use_deterministic_algorithms`. Slows training and "
            "packed variable-length attention may still prevent bitwise "
            "reproducibility."
        ),
    )
    compile: bool = Field(
        False,
        description=(
            "Compile the model with `torch.compile` (inductor backend), "
            "fusing RMSNorm/SwiGLU/rotary elementwise chains. Adds 1-3 min "
            "one-time warmup; see `compile_dynamic` with token-budget "
            "batching."
        ),
    )
    compile_mode: Literal[
        "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    ] = Field(
        "default",
        description=(
            "`torch.compile` mode. Avoid `reduce-overhead`: it needs static "
            "shapes/buffers (CUDA graphs)."
        ),
    )
    compile_dynamic: bool | None = Field(
        None,
        description=(
            "Shape specialization. `None` auto-detects after one recompile; "
            "`True` compiles one shape-agnostic graph up front (recommended "
            "with `dataloader.max_tokens`, since batch shapes vary); `False` "
            "forces static shapes, recompiling per distinct shape."
        ),
    )
    compile_fullgraph: bool = Field(
        False,
        description=(
            "Error on any graph break instead of falling back to eager. "
            "Diagnostic only."
        ),
    )
    dist_timeout_minutes: int = Field(
        90,
        gt=0,
        description=(
            "NCCL/Gloo collective timeout. Raise this for large first-time "
            "dataset cache builds (default 10 min can be too short)."
        ),
    )


class Trainer:
    """Orchestrates pretraining lifecycle using injected dependencies."""

    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_dataloader_provider: Callable[[int], torch.utils.data.DataLoader],
        config: TrainerConfig,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        loss_fn: torch.nn.Module | None = None,
        metric_logger: TrainingLogger | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.accelerator = accelerator
        self.is_main = self.accelerator.is_main_process
        self.device = self.accelerator.device  # Explicit device reference

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader_provider = train_dataloader_provider
        self.val_dataloader = val_dataloader
        # Track the epoch's prepared train dataloader so the previous epoch's
        # can be de-registered from the Accelerator
        self._prepared_train_dataloader: Any | None = None

        # Optional custom loss function (defaults to model's internal loss)
        self.loss_fn = loss_fn
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(self.device)

        # Saved alongside HF-format checkpoints so the tokenizer version used
        # to train a given checkpoint is never ambiguous.
        self.tokenizer = tokenizer

        self.config = config
        self.max_grad_norm = config.max_grad_norm
        self.output_dir = config.output_dir

        # A mismatch would silently train at the wrong effective batch size
        # instead of failing, since `accelerator.accumulate` uses its own setting.
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        if (
            self.accelerator.gradient_accumulation_steps
            != self.gradient_accumulation_steps
        ):
            raise ValueError(
                "Accelerator was built with gradient_accumulation_steps="
                f"{self.accelerator.gradient_accumulation_steps}, but TrainerConfig "
                f"requests {self.gradient_accumulation_steps}. Pass "
                "`gradient_accumulation_steps=config.trainer.gradient_accumulation_steps` "
                "when constructing the Accelerator."
            )

        # Injected metric logger: accumulates metrics on-device and flushes them
        # through `accelerator.log`, which forwards to any configured tracker
        self.metric_logger = metric_logger or TrainingLogger()
        self.metric_json_path = self.output_dir / "metrics.jsonl"

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.batches_completed_in_epoch: int = 0
        # Global_step of the most recent validation pass, so end-of-epoch
        # `evaluate()` can skip re-running/re-logging if `eval_steps` already
        # evaluated at the epoch's last step.
        self._last_eval_step: int = -1
        # Most recent step each checkpoint type was saved at, so the final
        # save in `train()` can skip re-saving if a `save_steps`/
        # `hf_save_steps` boundary already hit the last step.
        self._last_save_step: int = -1
        self._last_hf_save_step: int = -1

        self._register_signal_handlers()

        # Checkpointed alongside model/optimizer/scheduler state so step/epoch
        # counters in the logger survive resumes.
        self.accelerator.register_for_checkpointing(self.metric_logger)

        # Load checkpoint if provided
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def _register_signal_handlers(self) -> None:
        """Saves state and exits gracefully on cluster preemptions."""

        def handle_sigterm(signum: int, frame: FrameType | None) -> None:
            if self.is_main:
                logger.warning(
                    f"Signal {signum} received. Saving checkpoint before exit..."
                )
            self.save_checkpoint()
            self.accelerator.wait_for_everyone()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_sigterm)

    def _release_previous_train_dataloader(self, current: Any) -> None:
        """De-registers the prior epoch's train dataloader from the Accelerator.

        `prepare()` appends to private `accelerator._dataloaders` with no
        removal API; since a new train dataloader is built each epoch, that
        list would grow unbounded. Dropping the stale entry keeps it at a
        stable `[val, train]` so indices match across save/restore.
        """
        previous = self._prepared_train_dataloader
        self._prepared_train_dataloader = current
        if previous is None or previous is current:
            return

        registered = self.accelerator._dataloaders
        for index, dataloader in enumerate(registered):
            if dataloader is previous:
                del registered[index]
                break

        # Dropping the last reference triggers __del__, shutting down
        # persistent workers explicitly instead of relying on GC timing.
        iterator = previous._iterator
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()

    def train(self, num_epochs: int | None = None) -> None:
        """Main entry point iterating over epochs.

        If `config.max_steps` is set, it overrides `num_epochs`, requesting
        new epoch mixtures until the step budget is exhausted (HuggingFace
        Trainer semantics).
        """
        num_epochs = num_epochs or self.config.num_epochs
        max_steps = self.config.max_steps
        if self.is_main:
            if max_steps:
                logger.info(f"=== Starting Training Loop (max_steps={max_steps}) ===")
            else:
                logger.info(f"=== Starting Training Loop ({num_epochs} Epochs) ===")

        epoch_iter = (
            count(self.current_epoch)
            if max_steps
            else range(self.current_epoch, num_epochs)
        )

        # One continuous bar for the whole run (not per epoch), ticking per
        # *optimizer* step to match `max_steps`/`logging_steps`/`eval_steps`
        # units. `initial` seeds the correct position on resume.
        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            disable=not self.is_main,
            unit="step",
            desc="Training",
        )

        for epoch in epoch_iter:
            self.current_epoch = epoch

            if self.is_main:
                logger.info(f"Building training mixture for epoch {epoch}...")
            mixture_start = time.perf_counter()
            # Rank 0 builds/caches this epoch's mixture (e.g. a first-time
            # `valid_counts` scan) alone first, then other ranks then reuse
            # the same warm cache instead of recomputing it in parallel.
            with self.accelerator.local_main_process_first():
                train_dataloader = self.train_dataloader_provider(epoch)
            self._release_previous_train_dataloader(train_dataloader)

            # Record original length to maintain correct tqdm totals when resuming
            try:
                total_batches = len(train_dataloader)
                # Optimizer-step units (else `train/epoch` is off by the
                # accumulation factor); `ceil` since Accelerate also syncs a
                # trailing partial group.
                self.metric_logger.steps_per_epoch = max(
                    math.ceil(total_batches / self.gradient_accumulation_steps), 1
                )
            except TypeError:
                total_batches = None
                self.metric_logger.steps_per_epoch = 1.0

            if self.is_main:
                batches_msg = (
                    f"{total_batches:,} batches per rank"
                    if total_batches is not None
                    else "unknown length"
                )
                logger.info(
                    f"Training mixture for epoch {epoch} ready ({batches_msg}) in "
                    f"{time.perf_counter() - mixture_start:.1f}s. Training progress is "
                    "reported by the tqdm bar (stderr) and periodic metric logs."
                )

            # Refresh the bar's total each epoch since the mixture length can
            # shift slightly (e.g. under a score curriculum).
            if not max_steps and total_batches is not None:
                pbar.total = math.ceil(self.metric_logger.steps_per_epoch * num_epochs)

            # If resuming mid-epoch, skip already processed batches. A config
            # change can shorten the epoch, leaving the saved count out of
            # range; skipping past the end would yield an empty epoch.
            if self.batches_completed_in_epoch > 0:
                if (
                    total_batches is not None
                    and self.batches_completed_in_epoch >= total_batches
                ):
                    if self.is_main:
                        logger.warning(
                            f"Checkpoint resumes epoch {epoch} at batch "
                            f"{self.batches_completed_in_epoch}, but the epoch now "
                            f"has only {total_batches} batches. Restarting this "
                            "epoch from the beginning; some data will be seen twice."
                        )
                    self.batches_completed_in_epoch = 0
                else:
                    train_dataloader = self.accelerator.skip_first_batches(
                        train_dataloader, self.batches_completed_in_epoch
                    )

            self._train_one_epoch(train_dataloader, epoch, total_batches, pbar)
            # `evaluate()` skips its own pass/log flush if `eval_steps`
            # already hit this global_step. No per-epoch checkpoint here
            # (that saved a checkpoint every epoch regardless of
            # `save_steps`/`hf_save_steps`) -- checkpointing relies solely on
            # the step-based triggers plus the guaranteed final save below.
            self.evaluate()

            # Reset batch counter for the next epoch
            self.batches_completed_in_epoch = 0

            if max_steps and self.global_step >= max_steps:
                if self.is_main:
                    logger.info(f"Reached max_steps={max_steps}. Stopping training.")
                break

        pbar.close()

        # Guaranteed final checkpoints, unless a step-based boundary already
        # saved at this exact step.
        if self._last_save_step != self.global_step:
            self.save_checkpoint()
        if self._last_hf_save_step != self.global_step:
            self.save_hf_checkpoint()

        self.accelerator.end_training()

    def _train_one_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        total_batches: int | None,
        pbar: tqdm,
    ) -> None:
        """Manages the dataloader iteration and progress tracking."""
        self.model.train()
        pbar.set_description(f"Epoch {epoch}")

        # Mesured for the wandb dataloader/* metrics.
        data_stall_ms = 0.0
        data_start = time.perf_counter()
        for batch in dataloader:
            data_stall_ms += (time.perf_counter() - data_start) * 1000

            # Accelerate handles gradient accumulation contexts
            with self.accelerator.accumulate(self.model):
                loss, outputs = self.training_step(batch)
                grad_norm = self.optimizer_step()

            # Accumulate metrics on every micro-batch
            self.metric_logger.add_train_batch(batch, outputs, loss)

            self.batches_completed_in_epoch += 1

            if self.accelerator.sync_gradients:
                self.global_step += 1
                self.metric_logger.num_steps = self.global_step
                pbar.update(1)
                self.metric_logger.add_dataloader_stall(data_stall_ms)
                data_stall_ms = 0.0

                should_log = self.global_step % self.config.logging_steps == 0
                should_eval = (
                    self.config.eval_steps is not None
                    and self.global_step % self.config.eval_steps == 0
                )

                # Reduce frequency of device->host caused by .item()
                if not pbar.disable and (should_log or should_eval):
                    pbar.set_postfix(
                        {"loss": f"{loss.item():.4f}", "step": self.global_step}
                    )

                # Eval before logging, so a step where both boundaries
                # coincide produces one combined log row instead of two --
                # the second `log()` call would otherwise report fabricated
                # all-zero train metrics after the first resets those counters.
                if should_eval:
                    self._run_validation()

                if should_log or should_eval:
                    self.metric_logger.log(
                        grad_norm=grad_norm,
                        weight_sq_sum=self._compute_weight_sq_sum(),
                        optimizer=self.optimizer,
                        accelerator=self.accelerator,
                        json_path=self.metric_json_path,
                    )

                # Mid-epoch state checkpoint
                if (
                    self.config.save_steps
                    and self.global_step % self.config.save_steps == 0
                ):
                    self.save_checkpoint()

                # Mid-epoch HF format checkpoint
                if (
                    self.config.hf_save_steps
                    and self.global_step % self.config.hf_save_steps == 0
                ):
                    self.save_hf_checkpoint()

                # Stop mid-epoch once the step budget is exhausted, rather
                # than running the rest of this epoch's batches needlessly.
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

            data_start = time.perf_counter()

        self.metric_logger.num_epochs = epoch + 1

    def compute_loss(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Any]:
        """Calculates loss and returns `(loss, outputs)`. Extracted so it can be overridden.

        `outputs` must at minimum expose `.logits`, since it is passed to the
        metric logger for accuracy tracking.
        """
        if self.loss_fn is not None:
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            loss = self.loss_fn(outputs.logits, inputs["labels"])
            return loss, outputs

        outputs = model(**inputs)
        loss = outputs.loss
        return loss, outputs

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, Any]:
        """Processes a single batch: forward pass, loss calc, backward pass."""
        loss, outputs = self.compute_loss(self.model, batch)
        self.accelerator.backward(loss)
        return loss, outputs

    def optimizer_step(self) -> torch.Tensor | None:
        """Updates parameters and handles gradient clipping/accumulation."""
        grad_norm = None
        if self.accelerator.sync_gradients:
            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        # Ensure scheduler steps only when weights are updated
        if self.accelerator.sync_gradients:
            self.scheduler.step()

        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _compute_weight_sq_sum(self) -> torch.Tensor:
        """Sum of squared trainable weights, used to track weight-norm drift over training."""
        from torch.distributed.tensor import DTensor

        with torch.no_grad():
            local_shards = [
                p.to_local() if isinstance(p, DTensor) else p
                for p in self.model.parameters()
                if p.requires_grad
            ]
            return sum(p.float().pow(2).sum() for p in local_shards)

    def evaluate(self) -> None:
        """Runs validation and immediately flushes metrics.

        Used for the guaranteed end-of-epoch evaluation; skips entirely if a
        mid-epoch `eval_steps` boundary already evaluated at this exact
        `global_step`, avoiding a redundant pass and duplicate metrics row.
        """
        if self.val_dataloader is None or self._last_eval_step == self.global_step:
            return

        self._run_validation()

        # Flush right after each eval instead of at the next `logging_steps`
        # boundary, which would merge consecutive evals into one report or
        # drop the final epoch's eval entirely.
        self.metric_logger.log(
            grad_norm=torch.tensor(0.0, device=self.device),
            weight_sq_sum=self._compute_weight_sq_sum(),
            optimizer=self.optimizer,
            accelerator=self.accelerator,
            json_path=self.metric_json_path,
        )

    def _run_validation(self) -> None:
        """Runs the validation loop, accumulating metrics into the metric logger.

        Does not flush/log -- callers call `metric_logger.log(...)` either
        immediately (`evaluate()`) or merged with a coincident train log (see
        `_train_one_epoch`).
        """
        if self.val_dataloader is None:
            return

        self.model.eval()
        with torch.inference_mode():
            for batch in self.val_dataloader:
                loss, outputs = self.compute_loss(self.model, batch)
                self.metric_logger.add_eval_batch("val", batch, outputs, loss)
        self.model.train()

        self._last_eval_step = self.global_step
        if self.is_main:
            logger.info(
                f"[Eval Epoch {self.current_epoch}] evaluation batches processed."
            )

    def save_checkpoint(self, path: Path | None = None) -> None:
        """Saves weights, optimizer state, scheduler, and RNG trackers.

        Decoupled from `save_hf_checkpoint`: this large, resumable Accelerate
        state saves on its own `save_steps` schedule and never cascades into
        an HF checkpoint save -- callers needing both must trigger each
        explicitly.
        """
        ckpt_dir = (
            Path(path)
            if path
            else self.output_dir
            / f"checkpoint_epoch_{self.current_epoch}_step_{self.global_step}"
        )
        self.accelerator.wait_for_everyone()

        if self.is_main:
            logger.info(f"Saving training state checkpoint to {ckpt_dir}...")

        self.accelerator.save_state(str(ckpt_dir))

        # Save custom trainer metadata required to resume accurately
        if self.is_main:
            metadata = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "batches_completed": self.batches_completed_in_epoch,
            }
            torch.save(metadata, ckpt_dir / "trainer_state.pt")

        self._last_save_step = self.global_step
        self.accelerator.wait_for_everyone()
        self._rotate_checkpoints(
            self.output_dir, "checkpoint_epoch_*_step_*", self.config.save_total_limit
        )

    def save_hf_checkpoint(self, path: Path | None = None) -> None:
        """Saves a portable Hugging Face checkpoint (model + tokenizer).

        Kept separate from `save_checkpoint`'s resumable Accelerate state;
        meant for downstream `AutoModel.from_pretrained(...)` loading, and
        always bundles the tokenizer so it's unambiguous which one a
        checkpoint expects.
        """
        hf_dir = (
            Path(path)
            if path
            else self.output_dir / "hf_checkpoints" / f"checkpoint_{self.global_step}"
        )
        if self.is_main:
            hf_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(
            self.model, keep_torch_compile=False
        )
        # Collective under FSDP/DeepSpeed, so every rank must call it.
        state_dict = self.accelerator.get_state_dict(self.model)
        # `get_state_dict` unwraps with `keep_torch_compile=True`, so under
        # `trainer.compile` keys get an `_orig_mod.` prefix that won't load
        # into a plain AMPLIFYForMaskedLM.
        state_dict = {
            key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
        }

        # Makes the checkpoint self-contained: copies AMPLIFY source next to
        # the weights and records auto_map, so `from_pretrained(dir,
        # trust_remote_code=True)` works without this repo.
        unwrapped_model.config.auto_map = register_amplify_auto_classes()
        if self.tokenizer is not None:
            self.tokenizer.register_for_auto_class("AutoTokenizer")

        # `save_pretrained` does not rank-guard its safetensors writes, so calling
        # it on every rank races all ranks onto the same file.
        if self.is_main:
            unwrapped_model.save_pretrained(
                hf_dir,
                is_main_process=True,
                save_function=self.accelerator.save,
                state_dict=state_dict,
            )

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(hf_dir)
            else:
                logger.warning(
                    "No tokenizer was injected into the Trainer; HF checkpoint at "
                    f"{hf_dir} will not include tokenizer files."
                )

        self.accelerator.wait_for_everyone()

        self._last_hf_save_step = self.global_step
        self._rotate_checkpoints(
            self.output_dir / "hf_checkpoints",
            "checkpoint_*",
            self.config.hf_save_total_limit,
        )

    def _rotate_checkpoints(
        self, base_dir: Path, pattern: str, limit: int | None
    ) -> None:
        """Deletes the oldest checkpoints under `base_dir` beyond `limit`.

        Ranked by the trailing step number in the directory name (works for
        both `checkpoint_epoch_<E>_step_<S>` and `checkpoint_<S>`), oldest
        first, keeping the most recent. No-op if `limit` is unset or
        `base_dir` doesn't exist.
        """
        if limit is None or not self.is_main or not base_dir.exists():
            return

        def _step(p: Path) -> int:
            match = re.search(r"(\d+)$", p.name)
            return int(match.group(1)) if match else -1

        checkpoints = sorted(
            (p for p in base_dir.glob(pattern) if p.is_dir()), key=_step
        )
        excess = len(checkpoints) - limit
        for ckpt_dir in checkpoints[: max(excess, 0)]:
            logger.info(f"Removing old checkpoint {ckpt_dir} (limit={limit})")
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def load_checkpoint(self, path: Path) -> None:
        """Restores training state from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {path}")

        if self.is_main:
            logger.info(f"Resuming from checkpoint: {path}")

        self.accelerator.load_state(str(path))

        metadata_path = path / "trainer_state.pt"
        if metadata_path.exists():
            metadata = torch.load(metadata_path, map_location="cpu")
            self.current_epoch = metadata["epoch"]
            self.global_step = metadata["global_step"]
            self.batches_completed_in_epoch = metadata.get("batches_completed", 0)
        else:
            logger.warning(
                "trainer_state.pt not found. Epoch and step counters will start at 0."
            )
