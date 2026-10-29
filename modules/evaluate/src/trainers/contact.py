"""Fixed-protocol fitting for CASP contact predictors and independent controls."""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from time import monotonic
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.model.contact_attention import (
    ATTENTION_C,
    ATTENTION_HEAD_TYPE,
    ATTENTION_MAX_ITER,
    ATTENTION_PROTOCOL,
    ContactAttentionModel,
    _encoder_layers,
)
from modules.evaluate.src.model.contact_model import (
    BilinearHead,
    ContactPredictionModel,
)

if TYPE_CHECKING:
    from modules.evaluate.src.dataset.contact import ContactDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BilinearProtocol:
    """Effective fitting settings, not configurable generic trainer options."""

    learning_rate: float = 0.001
    weight_decay: float = 10.0
    batch_size: int = 20
    early_stopping_patience: int = 10
    optimizer: str = "adamw"
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    max_grad_norm: float = 0.0
    accumulation_steps: int = 1


BILINEAR_PROTOCOL = BilinearProtocol()


ContactEpoch = TypedDict(
    "ContactEpoch", {"epoch": int, "loss": float, "validation_P@L": float}
)
BilinearFitSettings = TypedDict(
    "BilinearFitSettings",
    {
        "seed": int,
        "best_epoch": int,
        "validation_P@L": float,
        "stopped_by_patience": bool,
        "history": list[ContactEpoch],
        "optimizer": str,
        "learning_rate": float,
        "matrix_weight_decay": float,
        "projection": int,
        "pair_weighting": str,
        "patience": int,
        "effective_config": dict[str, object],
    },
)


class AttentionFitSettings(TypedDict):
    seed: int
    predictor: str
    attention_protocol: str
    training_structures: int
    training_pairs: int
    positive_pairs: int
    feature_count: int
    penalty: str
    C: float
    solver: str
    max_iter: int
    iterations: int
    standardized: bool


class ZeroShotSettings(TypedDict):
    fitting: str


class ContactControlSettings(TypedDict):
    control: str
    protocol: str
    trunk_initialization: str
    trunk_seed: int
    head_seed: int | None
    development_features: str | None
    fit: BilinearFitSettings | AttentionFitSettings | ZeroShotSettings


def fit_contact_head(
    model: ContactPredictionModel,
    dataset: ContactDataset,
    features: dict[str, torch.Tensor],
    *,
    seed: int,
    max_epochs: int,
    device: torch.device,
) -> BilinearFitSettings:
    """One pair-weighted 20-protein AdamW step per epoch, validation patience 10."""
    from modules.evaluate.src.dataset.contact import candidate_pairs
    from modules.evaluate.src.metrics.contact_metrics import score_contact_unit

    training, validation = dataset.splits["train"], dataset.splits["validation"]
    protocol = BILINEAR_PROTOCOL
    if len(training) != protocol.batch_size or len(validation) != protocol.batch_size:
        raise ValueError(
            "Contact fitting requires 20 training and 20 validation structures"
        )
    if max_epochs <= protocol.early_stopping_patience:
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
                "weight_decay": protocol.weight_decay,
            },
            {"params": [model.head.bias], "weight_decay": 0.0},
        ],
        lr=protocol.learning_rate,
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
    history: list[ContactEpoch] = []
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
        if stale == protocol.early_stopping_patience:
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
    if stale != protocol.early_stopping_patience or best_state is None:
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
        "learning_rate": protocol.learning_rate,
        "matrix_weight_decay": protocol.weight_decay,
        "projection": 512,
        "pair_weighting": "equal_unordered",
        "patience": protocol.early_stopping_patience,
        "effective_config": {**asdict(protocol), "max_epochs": max_epochs},
    }


def fit_contact_attention_head(
    model: ContactAttentionModel,
    dataset: ContactDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    seed: int,
) -> AttentionFitSettings:
    """Fit the fixed L1 logistic head on every known unordered training pair."""
    from modules.evaluate.src.dataset.contact import candidate_pairs

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
