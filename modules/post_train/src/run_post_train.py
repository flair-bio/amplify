"""Dispatch validated post-training settings to a method-specific runner."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

# Must be set before importing runners that can initialize CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from modules.core.utils.config_loader import load_runtime_config_from_argv  # noqa: E402
from modules.post_train.src.contracts.config import PostTrainingConfig  # noqa: E402


def run(config: PostTrainingConfig) -> dict[str, Any]:
    if config.method == "latent_alignment":
        from modules.post_train.src.runners.latent_alignment import (
            run as run_latent_alignment,
        )

        return run_latent_alignment(config)
    raise ValueError(f"Unsupported post-training method: {config.method!r}")


def main(argv: Sequence[str] | None = None) -> None:
    config = load_runtime_config_from_argv(
        sys.argv[1:] if argv is None else argv,
        PostTrainingConfig,
    )
    run(config)


if __name__ == "__main__":
    main()
