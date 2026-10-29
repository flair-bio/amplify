"""Frozen-trunk models for CASP contact prediction."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase

from modules.evaluate.src.model.adapters import contact_positions

if TYPE_CHECKING:
    from modules.evaluate.src.dataset.contact import ContactChain

logger = logging.getLogger(__name__)


class BilinearHead(nn.Module):
    """Two bias-free projections, averaged directional logits and scalar bias."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.left = nn.Linear(dimension, 512, bias=False, dtype=torch.float32)
        self.right = nn.Linear(dimension, 512, bias=False, dtype=torch.float32)
        self.bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        directional = self.left(embeddings) @ self.right(embeddings).T
        return (directional + directional.T) / 2 + self.bias


class ContactPredictionModel(nn.Module):
    """Frozen backbone plus independently fitted 512-dimensional bilinear head."""

    def __init__(self, trunk: Any) -> None:
        super().__init__()
        self.trunk: Any = trunk.float().eval().requires_grad_(False)
        self.head = BilinearHead(int(trunk.config.hidden_size))

    @torch.no_grad()
    def extract(
        self, chain: ContactChain, tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        from modules.evaluate.src.dataset.contact import tokenize_contact_chain

        tokens = tokenize_contact_chain(chain, tokenizer)
        device = next(self.trunk.parameters()).device
        inputs = {
            "input_ids": torch.tensor([tokens.input_ids], device=device),
            "attention_mask": torch.tensor([tokens.attention_mask], device=device),
        }
        positions = torch.tensor([tokens.position_ids], device=device)
        if "position_ids" in inspect.signature(self.trunk.forward).parameters:
            inputs["position_ids"] = positions
        self.trunk.eval()
        with contact_positions(self.trunk, positions):
            output = self.trunk(**inputs)
        hidden = output.last_hidden_state
        if hidden.shape[:2] != inputs["input_ids"].shape:
            raise ValueError("Backbone changed the full-chain token layout")
        features = hidden[0, list(tokens.residue_token_indices)].float().cpu()
        if not torch.isfinite(features).all():
            raise ValueError("Contact embeddings must be finite")
        return features

    @torch.no_grad()
    def predict(
        self, chain: ContactChain, tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        features = self.extract(chain, tokenizer).to(self.head.bias.device)
        self.head.eval()
        return self.head(features).cpu()

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.trunk.save_pretrained(directory)
        torch.save(self.head.state_dict(), directory / "contact_head.pt")
        (directory / "contact_config.json").write_text(
            json.dumps(
                {
                    "protocol": "casp-contact-v1",
                    "predictor": "bilinear512",
                    "projection": 512,
                    "dimension": self.head.left.in_features,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ContactPredictionModel:
        directory = Path(directory)
        config = json.loads((directory / "contact_config.json").read_text())
        if config["protocol"] != "casp-contact-v1" or config["projection"] != 512:
            raise ValueError("Unsupported contact checkpoint protocol")
        if config.get("predictor", "bilinear512") != "bilinear512":
            raise ValueError("Unsupported bilinear contact checkpoint predictor")
        model = build_contact_model(str(directory))
        model.head.load_state_dict(
            torch.load(
                directory / "contact_head.pt", map_location="cpu", weights_only=True
            )
        )
        return model.eval()


def build_contact_model(
    model_id: str,
    predictor: str = "bilinear512",
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Any:
    if predictor == "attention_l1":
        from modules.evaluate.src.model.contact_attention import ContactAttentionModel

        return ContactAttentionModel(load_contact_trunk(model_id))
    if predictor == "categorical_jacobian":
        from modules.evaluate.src.model.contact_jacobian import (
            build_contact_jacobian_model,
        )

        if tokenizer is None:
            raise ValueError("The categorical Jacobian predictor requires a tokenizer")
        return build_contact_jacobian_model(model_id, tokenizer)
    if predictor != "bilinear512":
        raise ValueError(f"Unknown CASP contact predictor: {predictor}")
    return ContactPredictionModel(load_contact_trunk(model_id))


def load_contact_trunk(model_id: str) -> Any:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Masked-LM ESM checkpoints have no pooler; it is unused for residue features.
    kwargs: dict[str, Any] = (
        {"add_pooling_layer": False} if config.model_type == "esm" else {}
    )
    return AutoModel.from_pretrained(
        model_id, config=config, trust_remote_code=True, **kwargs
    )


def load_contact_model(directory: str | Path) -> Any:
    directory = Path(directory)
    config = json.loads((directory / "contact_config.json").read_text())
    if config.get("predictor", "bilinear512") == "attention_l1":
        from modules.evaluate.src.model.contact_attention import ContactAttentionModel

        return ContactAttentionModel.from_pretrained(directory)
    if config.get("predictor", "bilinear512") != "bilinear512":
        raise ValueError("Unsupported contact checkpoint predictor")
    return ContactPredictionModel.from_pretrained(directory)
