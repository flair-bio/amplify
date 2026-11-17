"""Explicit model construction shared across evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from sklearn.base import BaseEstimator
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from modules.evaluate.src.config.evaluation import LayerwiseConfig

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Predictor:
    """Live predictor resources, separate from training and test inputs."""

    model: nn.Module | BaseEstimator
    tokenizer: PreTrainedTokenizerBase | None
    trunk: nn.Module | None = None


def load_tokenizer(model_id: str, *, preserve_coordinates: bool = False) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if preserve_coordinates and getattr(tokenizer, "remove_ambiguous", False):
        tokenizer.remove_ambiguous = False
        tokenizer.init_kwargs["remove_ambiguous"] = False
    return tokenizer


def build_backbone(model_id: str, **kwargs: Any) -> nn.Module:
    return AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)


def build_masked_lm(model_id: str, **kwargs: Any) -> nn.Module:
    return AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True, **kwargs
    )


def build_predictor(
    model_id: str,
    task_type: str,
    num_labels: int,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    kwargs = dict(model_kwargs or {})
    if task_type in {"pseudo_perplexity", "calibration"}:
        return build_masked_lm(model_id, **kwargs)
    defaults = (
        {"problem_type": "regression"}
        if task_type == "sequence_regression"
        else {"problem_type": "multi_label_classification"}
        if task_type == "sequence_multilabel_classification"
        else {}
    )
    wrapper = (
        AutoModelForTokenClassification
        if task_type == "token_classification"
        else AutoModelForSequenceClassification
    )
    return wrapper.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
        **{**defaults, **kwargs},
    )


def build_probe(
    trunk: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    task_type: str,
    num_labels: int,
    *,
    embedding_size: int | None = None,
    layerwise_config: LayerwiseConfig | None = None,
) -> nn.Module:
    """Attach the configured head without making data preparation own models."""
    if layerwise_config is not None and layerwise_config.enabled:
        from modules.evaluate.src.model.layerwise import LayerwiseProbeModel

        return LayerwiseProbeModel(
            trunk, num_labels, task_type, layerwise_config, tokenizer=tokenizer
        )
    if embedding_size is not None:
        from modules.evaluate.src.model.heads import EmbeddingProbeHead

        return EmbeddingProbeHead(embedding_size, num_labels, task_type)
    return trunk
