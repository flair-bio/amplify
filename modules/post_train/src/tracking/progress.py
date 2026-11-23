"""Keep machine-readable metrics separate from terminal and W&B reporting."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

import torch
from tqdm.auto import tqdm

from modules.post_train.src.contracts.config import PostTrainingConfig
from modules.post_train.src.trainers.latent_alignment import capture_rng, restore_rng


def wandb_metrics(row: dict[str, Any]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}

    def flatten(prefix: str, values: Any) -> None:
        if isinstance(values, dict):
            for key, value in values.items():
                flatten(f"{prefix}/{key}", value)
        elif isinstance(values, (int, float)):
            result[prefix] = values

    for key, value in row.items():
        if key != "step":
            flatten("validation" if key == "validation" else f"train/{key}", value)
    return result


class ProgressLogger:
    def __init__(
        self, stream: TextIO, progress: Any, tracker: Any, device: torch.device
    ) -> None:
        self.stream = stream
        self.progress = progress
        self.tracker = tracker
        self.device = device
        self.validation_mlm: float | None = None

    def phase(self, description: str) -> None:
        self.progress.set_description_str(description)

    def log(self, row: dict[str, Any]) -> None:
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        if self.tracker is not None:
            state = capture_rng(self.device)
            try:
                self.tracker.log(wandb_metrics(row), step=row["step"])
            finally:
                restore_rng(state, self.device)
        if "validation" in row:
            self.validation_mlm = row["validation"]["mlm"]
        postfix = {
            key: f"{row[key]:.4f}" for key in ("total", "mlm", "latent") if key in row
        }
        if self.validation_mlm is not None:
            postfix["val_mlm"] = f"{self.validation_mlm:.4f}"
        self.phase("Post-training")
        self.progress.set_postfix(postfix, refresh=False)
        self.progress.update(row["step"] - self.progress.n)


@contextmanager
def track_run(
    config: PostTrainingConfig,
    output: Path,
    resolved: dict[str, Any],
    steps: int,
    device: torch.device,
) -> Iterator[ProgressLogger]:
    with (
        (output / "metrics.jsonl").open("w", buffering=1) as stream,
        tqdm(
            total=steps,
            desc="Starting",
            unit="update",
            dynamic_ncols=True,
            disable=not config.training.progress_bar,
        ) as progress,
    ):
        tracker = None
        options = config.wandb
        if options.enabled and options.mode != "disabled":
            state = capture_rng(device)
            try:
                import wandb

                directory = (options.dir or output / "wandb").expanduser().resolve()
                directory.mkdir(parents=True, exist_ok=True)
                tracker = wandb.init(
                    project=options.project,
                    entity=options.entity,
                    name=options.run_name or output.name,
                    tags=options.tags,
                    mode=options.mode,
                    dir=str(directory),
                    config=resolved,
                    settings=wandb.Settings(disable_code=True, disable_git=True),
                )
            finally:
                restore_rng(state, device)
        try:
            yield ProgressLogger(stream, progress, tracker, device)
        except BaseException:
            if tracker is not None:
                tracker.finish(exit_code=1)
            raise
        else:
            if tracker is not None:
                tracker.finish(exit_code=0)
