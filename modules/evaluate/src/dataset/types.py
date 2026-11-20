"""Typed evaluation inputs, separate from published results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from modules.evaluate.src.dataset.embeddings import EmbeddingBackend


@dataclass
class CachedFeatures:
    """Cached-representation backend and feature-extraction metadata."""

    backend: EmbeddingBackend


@dataclass
class LayerwiseInputs:
    """Token loaders for layerwise probe evaluation."""

    token_dataloaders: dict[str, DataLoader | None]


@dataclass
class FeatureRequest:
    """Scientific extraction requirements, independent of cache location/policy."""

    namespace: str
    identity: dict[str, Any]
    extract: Callable[[], dict[str, torch.Tensor]]
    shapes: dict[str, tuple[int, ...]]
    dtype: torch.dtype = torch.float32


@dataclass
class FeatureSet:
    values: dict[str, torch.Tensor]
    fingerprint: str


@dataclass
class TrainingData:
    """Development inputs; never owns a predictor or fitted control."""

    train_dataloader: DataLoader | None
    val_dataloader: DataLoader | None
    refit_dataloaders: dict[str, DataLoader] = field(default_factory=dict)
    refit_token_dataloaders: dict[str, DataLoader] = field(default_factory=dict)
    refit_random_truncate_by_split: dict[str, bool] = field(default_factory=dict)
    embedding: CachedFeatures | None = None
    layerwise: LayerwiseInputs | None = None
    validation_metric_groups: dict[str, Any] | None = None
    split: str = "default"


@dataclass
class TestData:
    """One evaluation view, separate from development inputs and model state."""

    test_dataloader: DataLoader | None
    embedding: CachedFeatures | None = None
    layerwise: LayerwiseInputs | None = None
    split: str = "default"
    baseline_context: dict[str, Any] | None = None
    test_original_positions: dict[str, list[int]] = field(default_factory=dict)
    test_metric_groups: dict[str, Any] = field(default_factory=dict)
    test_group_column: str | None = None
