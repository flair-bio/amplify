"""Evaluation config composition.

Task behavior is always selected explicitly (by a suite or a task overlay).
Dataset YAML files describe data only and never choose a task.

Layers are written in canonical namespaces; legacy keys are aliased by
``modules.evaluate.src.config.namespaces`` before the strict validation below.
"""

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from modules.core.utils.config_loader import ModelT, parse_config
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.config.namespaces import compose_canonical


def load_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> EvaluationConfig:
    return parse_config(compose_evaluation_config(path, overrides), EvaluationConfig)


def load_and_parse_evaluation(
    path: str | Path | Sequence[str | Path],
    model_cls: type[ModelT],
    overrides: Sequence[str] | None = None,
) -> ModelT:
    """Compose through the evaluate namespace mapping, then validate as ``model_cls``.

    Use this instead of the generic core ``load_and_parse`` whenever the YAML
    may use canonical or legacy evaluation namespaces.
    """
    return parse_config(compose_evaluation_config(path, overrides), model_cls)


def compose_evaluation_config(
    path: str | Path | Sequence[str | Path],
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Compose YAML files left-to-right with optional dotlist overrides.

    Each layer is canonicalized before merging (see ``config.namespaces``), so
    later layers win regardless of whether they use canonical or legacy keys.
    The result uses the internal legacy shape that ``EvaluationConfig`` validates.
    """
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    return compose_canonical(paths, overrides)
