import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType


class TrainingLogger:
    """Checkpointable metric logger for distributed training.

    Train and Eval metrics are accumulated purely on GPU across all ranks.
    A single all-reduce at log time synchronizes all counters across ranks.
    Zero .item() calls are made during either train or eval loops.
    """

    _STATE_KEYS = (
        "num_steps",
        "num_epochs",
        "num_batches_in_epoch",
        "num_samples",
        "num_tokens",
        "num_masked_tokens",
    )
    _TRAIN_KEYS = ("samples", "tokens", "pred", "loss_sum", "correct_sum")

    def __init__(self) -> None:
        self.num_steps = 0
        self.num_epochs = 0
        self.num_batches_in_epoch = 0
        self.num_samples = 0
        self.num_tokens = 0
        self.num_masked_tokens = 0
        self.steps_per_epoch: float = 1.0
        self._last_log_time = time.perf_counter()
        self._last_log_step = 0
        self._train: Dict[str, torch.Tensor] = {}
        self._eval: Dict[str, Dict[str, torch.Tensor]] = {}
        self._device = torch.device("cpu")
        # Limited to 1k in case logging_steps is ever disabled or too large.
        self._dataloader_stalls: deque[float] = deque(maxlen=1000)
        self._dataloader_stall_total_ms = 0.0
        self._reset()

    def _reset(self) -> None:
        self._dataloader_stalls.clear()
        self._dataloader_stall_total_ms = 0.0
        # Initialize flat on CPU, will migrate to device on first batch.
        self._train = {
            k: torch.tensor(0.0, dtype=torch.float64, device=self._device)
            for k in self._TRAIN_KEYS
        }
        self._eval.clear()

    def _ensure_device(self, device: torch.device) -> None:
        ## Cheap pointer/device comparison instead of reconstructing dict loops
        if self._device != device:
            self._device = device
            self._train = {k: v.to(device) for k, v in self._train.items()}

    def state_dict(self) -> Dict[str, int]:
        return {k: getattr(self, k) for k in self._STATE_KEYS}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        for k in self._STATE_KEYS:
            setattr(self, k, state.get(k, 0))
        self._last_log_time = time.perf_counter()
        self._last_log_step = self.num_steps

    def add_train_batch(
        self, batch: Dict[str, Any], output: Any, loss: torch.Tensor
    ) -> None:
        """Accumulate train metrics entirely on GPU.

        `loss` is taken explicitly (rather than read from `output.loss`) so
        that a custom `loss_fn` still works: in that path `output` comes from
        calling the model without labels, so `output.loss` is `None`.
        """
        labels = batch["labels"]
        self._ensure_device(labels.device)

        mask = labels.ne(-100)
        num_pred = mask.sum()
        t = self._train

        if "num_sequences" in batch:
            # Packing mode: collator provides exact counts.
            t["samples"] += batch["num_sequences"]
            t["tokens"] += batch["num_tokens"]
        else:
            t["samples"] += labels.shape[0]
            t["tokens"] += batch["attention_mask"].sum()

        t["pred"] += num_pred
        t["loss_sum"] += loss.detach() * num_pred
        t["correct_sum"] += (
            output.logits.detach().argmax(dim=-1).eq(labels).logical_and(mask).sum()
        )

    def add_dataloader_stall(self, stall_ms: float) -> None:
        """Record one optimizer step's wait on `next(dataloader)`, in ms."""
        self._dataloader_stalls.append(stall_ms)
        self._dataloader_stall_total_ms += stall_ms

    def add_eval_batch(
        self, split: str, batch: Dict[str, Any], output: Any, loss: torch.Tensor
    ) -> None:
        """Accumulate eval metrics purely on GPU without host synchronization.

        See `add_train_batch` for why `loss` is passed explicitly instead of
        read from `output.loss`.
        """
        labels = batch["labels"]
        device = labels.device
        mask = labels.ne(-100)
        num_pred = mask.sum()

        if split not in self._eval:
            self._eval[split] = {
                "pred": torch.tensor(0.0, dtype=torch.float64, device=device),
                "loss_sum": torch.tensor(0.0, dtype=torch.float64, device=device),
                "correct_sum": torch.tensor(0.0, dtype=torch.float64, device=device),
            }

        e = self._eval[split]
        e["pred"] += num_pred
        e["loss_sum"] += loss.detach() * num_pred
        e["correct_sum"] += (
            output.logits.detach().argmax(dim=-1).eq(labels).logical_and(mask).sum()
        )

    def log(
        self,
        grad_norm: torch.Tensor,
        weight_sq_sum: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        accelerator: Accelerator,
        json_path: Path,
    ) -> None:
        """Reduce accumulated metrics across ranks, log, and reset."""
        device = accelerator.device
        self._ensure_device(device)

        # Pack train + weight_sq_sum + eval tensors cleanly
        to_reduce: Dict[str, torch.Tensor] = {k: v for k, v in self._train.items()}
        to_reduce["weight_sq_sum"] = weight_sq_sum.to(
            device=device, dtype=torch.float64
        )

        to_reduce["dataloader_stall_ms"] = torch.tensor(
            self._dataloader_stall_total_ms, dtype=torch.float64, device=device
        )

        eval_splits = list(self._eval.keys())
        for name in eval_splits:
            for k, v in self._eval[name].items():
                to_reduce[f"{name}/{k}"] = v

        # Single fused all-reduce across all processes.
        keys = list(to_reduce.keys())
        n = accelerator.num_processes
        stalls = torch.zeros(n, len(self._dataloader_stalls), dtype=torch.float64)
        stalls[accelerator.process_index] = torch.tensor(
            self._dataloader_stalls, dtype=torch.float64
        )
        packed = torch.cat(
            [torch.stack(list(to_reduce.values())), stalls.flatten().to(device)]
        )
        reduced = accelerator.reduce(packed, reduction="sum").cpu()
        r = dict(zip(keys, reduced[: len(keys)].tolist()))
        stalls_by_rank = reduced[len(keys) :].numpy().reshape(n, -1)
        wsq = (
            r["weight_sq_sum"]
            if accelerator.distributed_type == DistributedType.FSDP
            else r["weight_sq_sum"] / n
        )

        # Update tracking states
        self.num_samples += int(r["samples"])
        self.num_tokens += int(r["tokens"])
        self.num_masked_tokens += int(r["pred"])

        now = time.perf_counter()
        elapsed_sec = max(now - self._last_log_time, 1e-12)
        steps_since_last_log = self.num_steps - self._last_log_step

        avg_loss = r["loss_sum"] / max(r["pred"], 1e-12)
        metrics: Dict[str, float] = {
            "train/epoch": self.num_steps / self.steps_per_epoch,
            "train/global_step": self.num_steps,
            "train/samples": self.num_samples,
            "train/tokens": self.num_tokens,
            "train/masked_tokens": self.num_masked_tokens,
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/weight_norm": math.sqrt(max(wsq, 0.0)),
        }
        # Skip these keys if no train batches were accumulated since the last flush.
        if r["pred"] > 0:
            metrics["train/loss"] = avg_loss
            metrics["train/perplexity"] = math.exp(
                min(avg_loss, 100)
            )  # Prevent overflow
            metrics["train/accuracy"] = r["correct_sum"] / max(r["pred"], 1e-12)
            metrics["train/grad_norm"] = float(grad_norm)
            metrics["train/steps_per_second"] = steps_since_last_log / elapsed_sec
            metrics["train/tokens_per_second"] = r["tokens"] / elapsed_sec

        if stalls_by_rank.size > 0:
            all_stalls = stalls_by_rank.ravel()
            rank_max = stalls_by_rank.max(axis=1)
            metrics["dataloader/iter_p50_ms"] = np.quantile(all_stalls, 0.5)
            metrics["dataloader/iter_p90_ms"] = np.quantile(all_stalls, 0.9)
            metrics["dataloader/iter_p99_ms"] = np.quantile(all_stalls, 0.99)
            # Slowest rank far above the average indicates a straggler.
            metrics["dataloader/iter_max_ms"] = rank_max.max()
            metrics["dataloader/slowest_rank"] = int(rank_max.argmax())
            metrics["dataloader/iter_step_pct"] = (
                100 * r["dataloader_stall_ms"] / (elapsed_sec * 1000 * n)
            )

        for name in eval_splits:
            pred = r[f"{name}/pred"]
            if pred <= 0:
                continue
            eloss = r[f"{name}/loss_sum"] / pred
            metrics[f"eval/{name}/loss"] = eloss
            metrics[f"eval/{name}/perplexity"] = math.exp(min(eloss, 100))
            metrics[f"eval/{name}/accuracy"] = r[f"{name}/correct_sum"] / pred

        # Hand off tracking telemetry to accelerator (asynchronous where possible)
        accelerator.log(metrics, step=self.num_steps)

        if accelerator.is_main_process:
            json_path.parent.mkdir(parents=True, exist_ok=True)

            # Standardizing float conversion for cleaner JSON dumps
            with json_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in metrics.items()}) + "\n")

        self._last_log_time = now
        self._last_log_step = self.num_steps
        self._reset()
