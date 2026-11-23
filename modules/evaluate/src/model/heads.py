"""
Pluggable prediction heads for the evaluation probe.

``EmbeddingProbeHead`` is a linear or MLP torch probe over precomputed
embeddings. ``"torch_linear"`` is a single :class:`torch.nn.Linear` layer trained by
backpropagation through
:func:`~modules.evaluate.src.trainers.torch.train_torch`. With
``steps.prepare.embedding_cache.enabled=True``, it uses the
standalone :class:`EmbeddingProbeHead`. Without the cache, it trains the
model's native head only when that head is itself a plain ``nn.Linear``.
Layerwise mode instead applies independent linear heads at selected frozen
hidden-state depths, using live or cached features. The native-head path
supports AMPLIFY sequence heads and all shipped token heads while
rejecting composite heads such as ESM's sequence-classification head.
``"mlp"`` remains the
existing native model-head path, and can also train an MLP probe on
frozen cached embeddings. Every classical
head type (``"sklearn_linear"``, ``"knn"``, ``"random_forest"``,
``"xgboost"``) is a
scikit-learn-compatible estimator, fit once (optionally swept over a
hyperparameter grid, scored on the validation split) directly on cached
pooled embeddings -- these have no gradient, so they use the classical
trainer. They therefore require
``steps.prepare.embedding_cache.enabled=True``: a classical estimator needs a
fixed-size feature vector per example, not a token sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

_FEATURE_STANDARDIZATION_CHUNK_SIZE = 65536
logger = logging.getLogger(__name__)


@dataclass
class _HeadOutput:
    """Minimal model-output contract shared by every probe head."""

    loss: torch.Tensor | None
    logits: torch.Tensor


def build_probe_layers(
    input_dim: int, hidden_size: int | None, output_dim: int, hidden_depth: int = 1
) -> tuple[nn.Module, nn.Module]:
    """Build a linear head or repeated hidden Linear-GELU layers plus output."""
    if hidden_size is None:
        return (nn.Identity(), nn.Linear(input_dim, output_dim))
    if hidden_depth < 1:
        raise ValueError("MLP hidden_depth must be positive.")
    layers: list[nn.Module] = []
    for index in range(hidden_depth):
        layers.extend(
            [
                nn.Linear(input_dim if index == 0 else hidden_size, hidden_size),
                nn.GELU(),
            ]
        )
    return (nn.Sequential(*layers), nn.Linear(hidden_size, output_dim))


class EmbeddingProbeHead(nn.Module):
    """A linear or MLP probe over cached or live frozen features.

    Cached probing uses this instead of the trunk and native task head;
    layerwise probing attaches one instance per selected depth. It returns
    the native wrappers' loss/logits contract so both paths share training
    and inference code.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        task_type: str,
        probe_hidden_size: int | None = None,
        standardize_features: bool = False,
        probe_hidden_depth: int = 1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.task_type = task_type
        self.embedding_size = hidden_size
        self.probe_hidden_size = probe_hidden_size
        self.probe_hidden_depth = probe_hidden_depth
        self.projection, self.classifier = build_probe_layers(
            hidden_size, probe_hidden_size, num_labels, probe_hidden_depth
        )
        self.standardize_features = standardize_features
        self.register_buffer(
            "feature_mean", torch.zeros(hidden_size) if standardize_features else None
        )
        self.register_buffer(
            "feature_std", torch.ones(hidden_size) if standardize_features else None
        )

    @torch.no_grad()
    def fit_standardization(
        self,
        embeddings: torch.Tensor,
        *,
        enabled: bool = True,
        row_mask: torch.Tensor | None = None,
        show_progress: bool = False,
    ) -> None:
        """Fit feature statistics on training rows; disabled heads keep raw inputs."""
        self.standardize_features = enabled
        if not enabled:
            self.feature_mean = self.feature_std = None
            return
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != self.embedding_size
        ):
            raise ValueError(
                f"Feature standardization requires nonempty training embeddings with shape (examples, {self.embedding_size})."
            )
        if row_mask is not None:
            if row_mask.ndim != 1 or row_mask.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    "Feature standardization row mask must have one value per training embedding."
                )
            row_mask = row_mask.to(device=embeddings.device, dtype=torch.bool)
        count = 0
        mean: torch.Tensor | None = None
        sum_squared_deviations: torch.Tensor | None = None
        show_progress = (
            show_progress and embeddings.shape[0] > _FEATURE_STANDARDIZATION_CHUNK_SIZE
        )
        with tqdm(
            total=embeddings.shape[0],
            desc="Standardizing cached probe features",
            unit="rows",
            disable=not show_progress,
        ) as progress:
            for start in range(
                0, embeddings.shape[0], _FEATURE_STANDARDIZATION_CHUNK_SIZE
            ):
                end = min(
                    start + _FEATURE_STANDARDIZATION_CHUNK_SIZE, embeddings.shape[0]
                )
                chunk = embeddings[start:end]
                if row_mask is not None:
                    chunk = chunk[row_mask[start:end]]
                if chunk.shape[0]:
                    chunk_std, chunk_mean = torch.std_mean(
                        chunk.float(), dim=0, correction=0
                    )
                    chunk_count = chunk.shape[0]
                    chunk_mean = chunk_mean.to(torch.float64)
                    chunk_m2 = chunk_std.to(torch.float64).square() * chunk_count
                    if mean is None or sum_squared_deviations is None:
                        mean = chunk_mean
                        sum_squared_deviations = chunk_m2
                        count = chunk_count
                    else:
                        combined_count = count + chunk_count
                        delta = chunk_mean - mean
                        sum_squared_deviations += chunk_m2 + delta.square() * (
                            count * chunk_count / combined_count
                        )
                        mean += delta * (chunk_count / combined_count)
                        count = combined_count
                progress.update(end - start)
        if mean is None or sum_squared_deviations is None:
            raise ValueError(
                f"Feature standardization requires nonempty training embeddings with shape (examples, {self.embedding_size})."
            )
        std = (sum_squared_deviations / count).sqrt().to(torch.float32)
        self.feature_mean = mean.to(self.classifier.weight)
        self.feature_std = std.clamp_min(1e-06).to(self.classifier.weight)

    def cache_identity(self) -> str:
        """Return the shape identity used by control-output caching."""
        return f"embedding_probe:{self.embedding_size}:{self.probe_hidden_size}:{self.probe_hidden_depth}:{self.num_labels}"

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Classify the final feature dimension at the head's device and dtype.

        Optional labels enable row-wise task loss. The layerwise wrapper handles
        padded live token losses separately, flattening logits and labels while
        ignoring special-token and padding targets.
        """
        inputs_embeds = inputs_embeds.to(
            device=self.classifier.weight.device, dtype=self.classifier.weight.dtype
        )
        if self.standardize_features:
            inputs_embeds = (inputs_embeds - self.feature_mean) / self.feature_std
        logits = self.classifier(self.projection(inputs_embeds))
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.task_type == "sequence_regression":
                loss = nn.functional.mse_loss(
                    logits.squeeze(-1), labels.to(logits.dtype)
                )
            elif self.task_type == "sequence_multilabel_classification":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.to(logits.dtype)
                )
            else:
                loss = nn.functional.cross_entropy(logits, labels)
        return _HeadOutput(loss=loss, logits=logits)


def validate_native_linear_head(model: nn.Module) -> None:
    """Require a plain native classifier for an uncached ``torch_linear`` run."""
    native_head = getattr(model, "classifier", None)
    if not isinstance(native_head, nn.Linear):
        raise ValueError(
            "head_type='torch_linear' without embedding_cache requires the loaded model's native classifier to be a single torch.nn.Linear. Enable steps.prepare.embedding_cache.enabled=True to use a standalone linear probe with this model."
        )
