"""Scientific identities and requests for fixed-protocol contact features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from modules.evaluate.src.artifacts.cache import weights_fingerprint
from modules.evaluate.src.dataset.contact import (
    PROTOCOL_VERSION,
    ContactDataset,
    ContactTokens,
    tokenize_contact_chain,
)
from modules.evaluate.src.dataset.types import FeatureRequest, FeatureSet
from modules.evaluate.src.model.contact_attention import (
    ATTENTION_PROTOCOL,
    ContactAttentionModel,
)
from modules.evaluate.src.model.contact_model import ContactPredictionModel
from modules.evaluate.src.model.factory import Predictor


def contact_development_feature_identity(
    dataset: ContactDataset,
    tokens: Sequence[ContactTokens],
    *,
    weights: str,
    backbone_config: Mapping[str, Any],
    tf32: bool,
    matmul_precision: str,
) -> dict[str, Any]:
    """Scientific identity of deterministic full-chain development features."""
    chains = [*dataset.splits["train"], *dataset.splits["validation"]]
    if len(tokens) != len(chains):
        raise ValueError(
            "Development feature identity requires one token row per chain"
        )
    return {
        "extraction": "casp-final-embeddings-v1",
        "weights": weights,
        "backbone_config": dict(backbone_config),
        "dtype": "float32",
        "tf32": tf32,
        "matmul_precision": matmul_precision,
        "chains": [
            (
                chain.split,
                chain.record_id,
                chain.alignment.sequence,
                token.input_ids,
                token.position_ids,
                token.residue_token_indices,
            )
            for chain, token in zip(chains, tokens)
        ],
    }


def contact_development_attention_fingerprint(
    dataset: ContactDataset,
    tokens: Sequence[ContactTokens],
    *,
    weights: str,
    backbone_config: Mapping[str, Any],
    tf32: bool,
    matmul_precision: str,
) -> str:
    """Fingerprint train-only attention inputs without retaining attention maps."""
    training = dataset.splits["train"]
    if len(tokens) != len(training):
        raise ValueError("Attention identity requires one token row per training chain")
    identity = {
        "protocol": ATTENTION_PROTOCOL,
        "development_fingerprint": dataset.development_fingerprint,
        "weights": weights,
        "backbone_config": dict(backbone_config),
        "precision": "float32",
        "tf32": tf32,
        "matmul_precision": matmul_precision,
        "chains": [
            (
                chain.record_id,
                chain.alignment.sequence,
                token.input_ids,
                token.position_ids,
                token.residue_token_indices,
            )
            for chain, token in zip(training, tokens)
        ],
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def contact_feature_requests(
    predictor: Predictor, dataset: ContactDataset
) -> dict[str, FeatureRequest]:
    model, tokenizer = (predictor.model, predictor.tokenizer)
    if not isinstance(model, ContactPredictionModel):
        return {}
    if dataset is None or tokenizer is None:
        raise ValueError("Contact fitting features require data and a tokenizer.")
    dataset = dataset
    chains = [*dataset.splits["train"], *dataset.splits["validation"]]
    tokens = [tokenize_contact_chain(chain, tokenizer) for chain in chains]
    identity = contact_development_feature_identity(
        dataset,
        tokens,
        weights=weights_fingerprint(model.trunk),
        backbone_config=json.loads(model.trunk.config.to_json_string()),
        tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision(),
    )

    def extract() -> dict[str, torch.Tensor]:
        return {
            str(index): model.extract(chain, tokenizer)
            for index, chain in enumerate(chains)
        }

    return {
        "development": FeatureRequest(
            namespace="contact-development",
            identity=identity,
            extract=extract,
            shapes={
                str(index): (
                    len(chain.alignment.clean_sequence),
                    model.head.left.in_features,
                )
                for index, chain in enumerate(chains)
            },
        )
    }


def contact_fitting_identity(
    predictor: Predictor, dataset: ContactDataset, features: dict[str, FeatureSet]
) -> dict[str, Any]:
    model = predictor.model
    if dataset is None or predictor.tokenizer is None:
        raise ValueError("Contact fitting identity requires data and a tokenizer.")
    dataset = dataset
    if isinstance(model, ContactPredictionModel):
        key = features["development"].fingerprint
        protocol = PROTOCOL_VERSION
    elif isinstance(model, ContactAttentionModel):
        key = contact_development_attention_fingerprint(
            dataset,
            [
                tokenize_contact_chain(chain, predictor.tokenizer)
                for chain in dataset.splits["train"]
            ],
            weights=weights_fingerprint(model.trunk),
            backbone_config=json.loads(model.trunk.config.to_json_string()),
            tf32=torch.backends.cuda.matmul.allow_tf32,
            matmul_precision=torch.get_float32_matmul_precision(),
        )
        protocol = ATTENTION_PROTOCOL
    else:
        return {}
    return {
        "development_fingerprint": dataset.development_fingerprint,
        "development_features": key,
        "protocol": protocol,
    }
