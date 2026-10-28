"""Frozen all-layer attention features and the fixed CASP logistic head."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from torch import nn
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.model.contact_model import (
    contact_positions,
    load_contact_trunk,
)

if TYPE_CHECKING:
    from modules.evaluate.src.tasks.contact_prediction import (
        ContactChain,
        ContactDataset,
    )

logger = logging.getLogger(__name__)

ATTENTION_HEAD_TYPE = "attention_l1"
ATTENTION_PROTOCOL = "casp-attention-l1-v1"
ATTENTION_MAX_ITER = 10_000
ATTENTION_C = 0.15


def _encoder_layers(trunk: Any) -> list[nn.Module]:
    if hasattr(trunk, "layers"):
        layers = list(trunk.layers)
    elif hasattr(trunk, "encoder") and hasattr(trunk.encoder, "layer"):
        layers = list(trunk.encoder.layer)
    else:
        raise ValueError(f"Cannot locate attention layers on {type(trunk).__name__}")
    if not layers:
        raise ValueError("Attention predictor requires at least one encoder layer")
    return layers


def _attention_modules(layers: list[nn.Module]) -> list[nn.Module]:
    modules = []
    for layer in layers:
        attention = getattr(layer, "attention", None)
        if attention is not None:
            modules.append(getattr(attention, "self", attention))
        elif hasattr(layer, "self_attn"):
            modules.append(layer.self_attn)
        else:
            modules.append(layer)
    return modules


def _apc_features(
    attention: torch.Tensor, residue_indices: list[int], *, softmax: bool
) -> torch.Tensor:
    attention = attention[0].float().cpu()
    if softmax:
        attention = attention.softmax(dim=-1)
    if not torch.isfinite(attention).all() or torch.any(attention < 0):
        raise ValueError("Attention probabilities must be finite and nonnegative")
    row_sums = attention.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.002, rtol=0):
        raise ValueError("Backbone attention maps must be normalized probabilities")
    attention = attention[:, residue_indices][:, :, residue_indices]

    values = attention + attention.transpose(-1, -2)
    row_sum = values.sum(dim=-1, keepdim=True)
    col_sum = values.sum(dim=-2, keepdim=True)
    total = values.sum(dim=(-1, -2), keepdim=True)
    features = values - row_sum * col_sum / total
    if not torch.isfinite(features).all():
        raise ValueError("APC-corrected attention features must be finite")
    return features


def _extract_attention_features(
    trunk: Any,
    chain: ContactChain,
    tokenizer: PreTrainedTokenizerBase,
    *,
    pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
    weights: np.ndarray | None = None,
) -> torch.Tensor:
    from modules.evaluate.src.tasks.contact_prediction import tokenize_contact_chain

    if (pair_indices is None) == (weights is None):
        raise ValueError("Provide either candidate pairs or fitted attention weights")
    tokens = tokenize_contact_chain(chain, tokenizer)
    device = next(trunk.parameters()).device
    inputs = {
        "input_ids": torch.tensor([tokens.input_ids], device=device),
        "attention_mask": torch.tensor([tokens.attention_mask], device=device),
        "output_attentions": True,
    }
    positions = torch.tensor([tokens.position_ids], device=device)
    if "position_ids" in inspect.signature(trunk.forward).parameters:
        inputs["position_ids"] = positions

    layers = _encoder_layers(trunk)
    num_heads = int(trunk.config.num_attention_heads)
    requires_softmax = getattr(trunk.config, "model_type", None) == "amplify"
    residue_indices = list(tokens.residue_token_indices)
    pair_tensors = (
        tuple(torch.as_tensor(index, dtype=torch.long) for index in pair_indices)
        if pair_indices is not None
        else None
    )
    if pair_tensors is not None and pair_tensors[0].shape != pair_tensors[1].shape:
        raise ValueError("Candidate pair indices must have matching shapes")
    if weights is not None and weights.shape != (len(layers) * num_heads,):
        raise ValueError("Attention weight count differs from the backbone layers")

    collected: list[torch.Tensor] = []
    logits = torch.zeros(
        (len(residue_indices), len(residue_indices)), dtype=torch.float64
    )
    layer_count = 0

    def reduce_attention(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        nonlocal layer_count
        if not isinstance(output, tuple):
            raise TypeError("Backbone layer did not return attention probabilities")
        result = list(output)
        attention_indices = [
            index
            for index, value in enumerate(result)
            if isinstance(value, torch.Tensor)
            and value.ndim == 4
            and value.shape[1] == num_heads
            and value.shape[-1] == value.shape[-2]
        ]
        if len(attention_indices) != 1:
            raise ValueError("Backbone layer returned an incompatible attention map")
        attention_index = attention_indices[0]
        features = _apc_features(
            result[attention_index], residue_indices, softmax=requires_softmax
        )
        if pair_tensors is not None:
            collected.append(features[:, pair_tensors[0], pair_tensors[1]])
        else:
            if weights is None:
                raise ValueError("Missing fitted attention weights")
            start = layer_count * num_heads
            layer_weights = torch.as_tensor(
                weights[start : start + num_heads], dtype=torch.float64
            )
            logits.add_(torch.einsum("h,hij->ij", layer_weights, features.double()))
        layer_count += 1
        result[attention_index] = features.new_empty((0,))
        return tuple(result)

    handles = [
        module.register_forward_hook(reduce_attention)
        for module in _attention_modules(layers)
    ]
    try:
        trunk.eval()
        with torch.inference_mode(), contact_positions(trunk, positions):
            trunk(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    if layer_count != len(layers):
        raise ValueError("Backbone did not return attention maps from every layer")
    if pair_tensors is not None:
        return torch.cat(collected, dim=0)
    return logits


class ContactAttentionModel(nn.Module):
    """Frozen trunk, all-layer/head APC features, and fitted L1 logistic head."""

    def __init__(self, trunk: Any) -> None:
        super().__init__()
        trunk.set_attn_implementation("eager")
        self.trunk: Any = trunk.float().eval().requires_grad_(False)
        self.classifier: LogisticRegression | None = None

    @torch.no_grad()
    def extract(
        self,
        chain: ContactChain,
        tokenizer: PreTrainedTokenizerBase,
        pair_indices: tuple[np.ndarray, np.ndarray],
    ) -> torch.Tensor:
        return _extract_attention_features(
            self.trunk, chain, tokenizer, pair_indices=pair_indices
        )

    @torch.no_grad()
    def predict(
        self, chain: ContactChain, tokenizer: PreTrainedTokenizerBase
    ) -> torch.Tensor:
        if self.classifier is None:
            raise RuntimeError("Attention contact model has not been fitted")
        weights = self.classifier.coef_[0]
        if weights.shape != (
            len(_encoder_layers(self.trunk))
            * int(self.trunk.config.num_attention_heads),
        ):
            raise ValueError("Attention layer/head count differs from the fitted head")
        logits = _extract_attention_features(
            self.trunk, chain, tokenizer, weights=weights
        )
        logits += float(self.classifier.intercept_[0])
        return torch.from_numpy(expit(logits.numpy()).astype(np.float32, copy=False))

    def save_pretrained(self, directory: str | Path) -> None:
        if self.classifier is None:
            raise RuntimeError("Cannot save an unfitted attention contact model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.trunk.save_pretrained(directory)
        joblib.dump(self.classifier, directory / "attention_head.joblib")
        (directory / "contact_config.json").write_text(
            json.dumps(
                {
                    "protocol": "casp-contact-v1",
                    "predictor": ATTENTION_HEAD_TYPE,
                    "attention_protocol": ATTENTION_PROTOCOL,
                    "features": "all_layers_heads_sum_symmetrized_apc",
                    "C": ATTENTION_C,
                    "solver": "liblinear",
                    "standardized": False,
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> ContactAttentionModel:
        directory = Path(directory)
        config = json.loads((directory / "contact_config.json").read_text())
        if (
            config.get("protocol") != "casp-contact-v1"
            or config.get("predictor") != ATTENTION_HEAD_TYPE
            or config.get("attention_protocol") != ATTENTION_PROTOCOL
        ):
            raise ValueError("Unsupported contact attention checkpoint protocol")
        model = cls(load_contact_trunk(str(directory)))
        model.classifier = joblib.load(directory / "attention_head.joblib")
        if (
            model.classifier.l1_ratio != 1.0
            or model.classifier.C != ATTENTION_C
            or model.classifier.solver != "liblinear"
        ):
            raise ValueError("Unsupported contact attention head protocol")
        return model.eval()


def fit_contact_attention_head(
    model: ContactAttentionModel,
    dataset: ContactDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit the fixed L1 logistic head on every known unordered training pair."""
    from modules.evaluate.src.tasks.contact_prediction import candidate_pairs

    training = dataset.splits["train"]
    if len(training) != 20:
        raise ValueError("Contact attention fitting requires 20 training structures")
    feature_count = len(_encoder_layers(model.trunk)) * int(
        model.trunk.config.num_attention_heads
    )
    pairs_by_chain = []
    total_pairs = 0
    for chain in training:
        if not chain.units[0].labels_available:
            raise ValueError("All contact training structures require labels")
        pairs = candidate_pairs(chain.units[0], chain.alignment, minsep=6)
        if not len(pairs.i):
            raise ValueError(f"{chain.record_id}: no eligible training pairs")
        pairs_by_chain.append((chain, pairs))
        total_pairs += len(pairs.i)

    X = np.empty((total_pairs, feature_count), dtype=np.float32)
    y = np.empty(total_pairs, dtype=np.int64)
    offset = 0
    for index, (chain, pairs) in enumerate(pairs_by_chain, start=1):
        values = model.extract(chain, tokenizer, (pairs.i, pairs.j))
        if (
            values.shape != (feature_count, len(pairs.i))
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"{chain.record_id}: invalid attention pair features")
        end = offset + len(pairs.i)
        X[offset:end] = values.T.numpy()
        y[offset:end] = pairs.labels
        logger.info(
            "Extracted training attention features %d/%d | record=%s | pairs=%d",
            index,
            len(pairs_by_chain),
            chain.record_id,
            len(pairs.i),
        )
        offset = end

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Attention logistic fitting requires positive and negative pairs"
        )
    classifier = LogisticRegression(
        l1_ratio=1.0,
        C=ATTENTION_C,
        solver="liblinear",
        random_state=seed,
        max_iter=ATTENTION_MAX_ITER,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(X, y)
    model.classifier = classifier
    logger.info(
        "Fitted CASP attention head | seed=%d | structures=%d | pairs=%d | "
        "features=%d | C=%.2f | solver=liblinear | iterations=%d",
        seed,
        len(training),
        len(y),
        X.shape[1],
        ATTENTION_C,
        int(classifier.n_iter_[0]),
    )
    return {
        "seed": seed,
        "predictor": ATTENTION_HEAD_TYPE,
        "attention_protocol": ATTENTION_PROTOCOL,
        "training_structures": len(training),
        "training_pairs": len(y),
        "positive_pairs": int(y.sum()),
        "feature_count": int(X.shape[1]),
        "penalty": "l1",
        "C": ATTENTION_C,
        "solver": "liblinear",
        "max_iter": ATTENTION_MAX_ITER,
        "iterations": int(classifier.n_iter_[0]),
        "standardized": False,
    }


def contact_development_attention_fingerprint(
    model: ContactAttentionModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: ContactDataset,
) -> str:
    """Fingerprint train-only inputs without retaining attention maps."""
    from modules.evaluate.src.tasks.contact_prediction import (
        tokenize_contact_chain,
    )
    from modules.evaluate.src.utils.embed import weights_fingerprint

    training = dataset.splits["train"]
    token_rows = [tokenize_contact_chain(chain, tokenizer) for chain in training]
    identity = {
        "protocol": ATTENTION_PROTOCOL,
        "dataset_fingerprint": dataset.fingerprint,
        "weights": weights_fingerprint(model.trunk),
        "backbone_config": json.loads(model.trunk.config.to_json_string()),
        "precision": "float32",
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "chains": [
            (
                chain.record_id,
                chain.alignment.sequence,
                tokens.input_ids,
                tokens.position_ids,
                tokens.residue_token_indices,
            )
            for chain, tokens in zip(training, token_rows)
        ],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return key
