"""Frozen-trunk models and fitting for CASP contact predictors."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from modules.evaluate.src.tasks.contact_prediction import (
        ContactChain,
        ContactDataset,
    )

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


@contextmanager
def contact_positions(model: Any, positions: torch.Tensor) -> Iterator[None]:
    """Use native rotary formulas, growing caches or temporarily selecting gaps."""
    # Current AMPLIFY honors position_ids but its precomputed tables are bounded.
    for module in model.modules():
        if all(
            hasattr(module, name) for name in ("rope_cos", "rope_sin", "_build_tables")
        ):
            length = int(positions.max()) + 1
            if length > module.max_position_embeddings:
                old_cos, old_sin = module.rope_cos, module.rope_sin
                module.max_position_embeddings = length
                module._build_tables(positions.device)
                if old_cos is not None and (
                    not torch.equal(old_cos, module.rope_cos[: len(old_cos)])
                    or not torch.equal(old_sin, module.rope_sin[: len(old_sin)])
                ):
                    raise ValueError(
                        "Native rotary cache growth changed existing positions"
                    )
    rotary = [
        module
        for module in model.modules()
        if all(
            hasattr(module, name)
            for name in ("_update_cos_sin_tables", "_seq_len_cached")
        )
    ]
    saved = []
    try:
        # HF ESM ignores position_ids in rotary attention; supply indexed native tables.
        for module in rotary:
            saved.append(
                (module, module._seq_len_cached, module._cos_cached, module._sin_cached)
            )
            dummy = torch.empty(
                (1, 1, int(positions.max()) + 1, 1), device=positions.device
            )
            cos, sin = module._update_cos_sin_tables(dummy, seq_dimension=-2)
            module._cos_cached = cos[0, 0][positions].unsqueeze(1)
            module._sin_cached = sin[0, 0][positions].unsqueeze(1)
            module._seq_len_cached = positions.shape[1]
        if hasattr(model, "freqs_cis"):
            table = model.freqs_cis
            length = int(positions.max()) + 1
            if length > len(table):
                source = inspect.getmodule(type(model))
                if source is None or not hasattr(source, "precompute_freqs_cis"):
                    raise ValueError("Cannot grow native rotary cache")
                table = source.precompute_freqs_cis(
                    model.config.hidden_size // model.config.num_attention_heads, length
                ).to(table.device)
                if not torch.equal(model.freqs_cis, table[: len(model.freqs_cis)]):
                    raise ValueError(
                        "Native rotary cache growth changed existing positions"
                    )
            model.freqs_cis = table.to(positions.device)[positions[0]]
            try:
                yield
            finally:
                model.freqs_cis = table
        else:
            gapped = not torch.equal(
                positions[0], torch.arange(positions.shape[1], device=positions.device)
            )
            if (
                gapped
                and not rotary
                and "position_ids" not in inspect.signature(model.forward).parameters
            ):
                raise ValueError("Backbone cannot consume original gapped positions")
            yield
    finally:
        for module, length, cos, sin in saved:
            module._seq_len_cached, module._cos_cached, module._sin_cached = (
                length,
                cos,
                sin,
            )


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
        from modules.evaluate.src.tasks.contact_prediction import tokenize_contact_chain

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


def fit_contact_head(
    model: ContactPredictionModel,
    dataset: ContactDataset,
    features: dict[str, torch.Tensor],
    *,
    seed: int,
    max_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """One pair-weighted 20-protein AdamW step per epoch, validation patience 10."""
    from modules.evaluate.src.metrics.contact_metrics import score_contact_unit
    from modules.evaluate.src.tasks.contact_prediction import candidate_pairs

    training, validation = dataset.splits["train"], dataset.splits["validation"]
    if len(training) != 20 or len(validation) != 20:
        raise ValueError(
            "Contact fitting requires 20 training and 20 validation structures"
        )
    if max_epochs < 11:
        raise ValueError("Contact epoch limit must allow validation patience 10")
    pairs = [candidate_pairs(chain.units[0], chain.alignment) for chain in training]
    if any(not len(pair.i) for pair in pairs):
        raise ValueError("Every training structure requires eligible contact pairs")
    if any(not chain.units[0].labels_available for chain in (*training, *validation)):
        raise ValueError("All development structures require available labels")
    # Backbone loading/extraction must not affect head initialization randomness.
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model.head = BilinearHead(model.head.left.in_features).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [model.head.left.weight, model.head.right.weight],
                "weight_decay": 10.0,
            },
            {"params": [model.head.bias], "weight_decay": 0.0},
        ],
        lr=0.001,
    )
    total = sum(len(pair.i) for pair in pairs)
    logger.info(
        "Starting contact-head fit | seed=%d | train_structures=%d | "
        "validation_structures=%d | training_pairs=%d | max_epochs=%d | "
        "patience=10 | lr=0.001 | matrix_weight_decay=10",
        seed,
        len(training),
        len(validation),
        total,
        max_epochs,
    )
    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
    history = []
    fit_started = monotonic()
    for epoch in range(1, max_epochs + 1):
        epoch_started = monotonic()
        model.head.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for chain, pair in zip(training, pairs):
            embeddings = features[chain.record_id].to(
                device=device, dtype=torch.float32
            )
            logits = model.head(embeddings)[pair.i, pair.j]
            truth = torch.as_tensor(pair.labels, device=device, dtype=torch.float32)
            loss = (
                F.binary_cross_entropy_with_logits(logits, truth, reduction="sum")
                / total
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite contact training loss")
            loss.backward()
            loss_sum += float(loss.detach())
        optimizer.step()
        model.head.eval()
        with torch.no_grad():
            scores = [
                score_contact_unit(
                    model.head(features[c.record_id].to(device)).cpu().numpy(),
                    c.units[0],
                    c.alignment,
                )["P@L"]
                for c in validation
            ]
        if any(value is None for value in scores):
            raise ValueError(
                "Every validation structure requires eligible contact pairs"
            )
        value = float(np.mean(scores))
        history.append({"epoch": epoch, "loss": loss_sum, "validation_P@L": value})
        if value > best_score:
            best_score, best_epoch, stale = value, epoch, 0
            best_state = deepcopy(model.head.state_dict())
            improved = True
        else:
            stale += 1
            improved = False
        logger.info(
            "Contact fit epoch %d/%d | train BCE=%.6f | validation mean P@L=%.6f | "
            "best=%s (epoch %d, %.6f) | patience=%d/10 | elapsed=%.1fs",
            epoch,
            max_epochs,
            loss_sum,
            value,
            "improved" if improved else "unchanged",
            best_epoch,
            best_score,
            stale,
            monotonic() - epoch_started,
        )
        if stale == 10:
            logger.info(
                "Contact fit early stopping at epoch %d/%d; restoring best head "
                "from epoch %d (validation mean P@L=%.6f, total fit time=%.1fs)",
                epoch,
                max_epochs,
                best_epoch,
                best_score,
                monotonic() - fit_started,
            )
            break
    if stale != 10 or best_state is None:
        raise RuntimeError(
            "Contact fit hit max_epochs before validation patience 10; raise the safety limit"
        )
    model.head.load_state_dict(best_state)
    model.head.eval()
    logger.info(
        "Contact fit complete | best_epoch=%d | validation mean P@L=%.6f | "
        "epochs=%d | restored_best_head=true",
        best_epoch,
        best_score,
        len(history),
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_P@L": best_score,
        "stopped_by_patience": True,
        "history": history,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "matrix_weight_decay": 10.0,
        "projection": 512,
        "pair_weighting": "equal_unordered",
        "patience": 10,
    }
