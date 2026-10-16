"""OmegaConf-backed config loading pipeline.

Full pipeline:
1. OmegaConf loads and merges one or more yaml files left to right.
2. CLI dotlist overrides are merged on top.
3. Pydantic validates the final merged dict against the provided schema.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load and merge one or more yaml config files, then apply CLI overrides."""
    paths = [path] if isinstance(path, (str, Path)) else path

    if not paths:
        raise ValueError("At least one config. path must be provided.")

    cfg = OmegaConf.merge(*[OmegaConf.load(str(p)) for p in paths])

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    return cast(DictConfig, cfg)


def parse_config(cfg: DictConfig, model_cls: type[ModelT]) -> ModelT:
    """Validate a raw OmegaConf dict. against a Pydantic model."""
    raw_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return model_cls.model_validate(raw_dict)


def load_and_parse(
    path: str | Path | Sequence[str | Path],
    model_cls: type[ModelT],
    overrides: Sequence[str] | None = None,
) -> ModelT:
    """Load, merge, and validate a configuration in one step.

    Example:
        cfg = load_and_parse("config.yaml", DataPipelineConfig)
    """
    cfg = load_config(path=path, overrides=overrides)
    return parse_config(cfg, model_cls=model_cls)


def load_runtime_config_from_argv(
    argv: Sequence[str], model_cls: type[ModelT]
) -> ModelT:
    """Load config from CLI args: ``<script>.py <config.yaml> [<override.yaml> ...] [key=value ...]``."""
    # OmegaConf-style dotlist overrides (key=value) vs. plain YAML config paths.
    config_paths = [a for a in argv if "=" not in a]
    overrides = [a for a in argv if "=" in a]

    if not config_paths:
        print(
            "Usage: <script>.py <config.yaml> [<override.yaml> ...] [key=value ...]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    logger.info("Loading config from: %s", config_paths)
    return load_and_parse(
        path=config_paths, model_cls=model_cls, overrides=overrides or None
    )
