"""Supervised per-assay evaluation: pool frozen-trunk embeddings, fit a linear
probe per CV fold, evaluate out-of-fold predictions against held-out variants.

Splits are either an official ProteinGym fold column (``fold_random_5``,
``fold_modulo_5``, ``fold_contiguous_5``) sourced alongside the assay, or a
freshly generated k-fold split when no official column is available/wanted.

``probe_type="linear"`` reuses the main pipeline's ``sklearn_linear`` head
(``model/heads.py``'s ``build_estimator``/``sweep_classical_head``: Ridge or
LogisticRegression, standardized internally, with the same C/alpha task
filtering used by ``TuneStep``) instead of a standalone implementation.
``probe_type="mlp"`` is a two-layer ``Linear-GELU-Linear`` torch MLP trained
by Adam (the same shape as ``EmbeddingProbeHead``'s projection+classifier),
and ``probe_type="torch_linear"`` is a single ``torch.nn.Linear`` layer
trained by Adam; both fit on pooled embeddings standardized (zero mean, unit
variance) per fold so regularization strength is comparable across trunks
with different embedding scales. Passing ``search_space`` sweeps a grid of
probe hyperparameters, scored on an inner holdout carved out of each fold's
training rows, then refits the winning hyperparameters on the fold's full
training rows.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.model.heads import (
    HeadType,
    build_estimator,
    build_probe_layers,
    sweep_classical_head,
)
from modules.evaluate.src.utils.embed import pool_hidden_states

logger = logging.getLogger(__name__)

SplitStrategy = Literal[
    "fold_random_5",
    "fold_modulo_5",
    "fold_contiguous_5",
    "custom_fold_5",
    "custom_kfold",
]
Task = Literal["regression", "classification"]
ProbeType = Literal["linear", "mlp", "torch_linear"]

# The three official ProteinGym CV schemes, averaged equally per assay by the
# official supervised leaderboard (performance_DMS_supervised_benchmarks.py).
OFFICIAL_CV_SCHEMES: tuple[str, ...] = (
    "fold_random_5",
    "fold_modulo_5",
    "fold_contiguous_5",
)

# Inner holdout fraction carved out of a fold's training rows when tuning.
_TUNING_VAL_FRACTION = 0.2


_TORCH_PROBE_LOSS: dict[Task, Any] = {
    "regression": nn.functional.mse_loss,
    "classification": nn.functional.binary_cross_entropy_with_logits,
}


class _TorchProbe:
    """A torch probe trained by Adam: one ``nn.Linear`` layer, or (with
    ``hidden_size`` set) a ``Linear-GELU-Linear`` MLP the same shape as
    ``EmbeddingProbeHead``'s projection+classifier. ``task="regression"``
    trains on MSE and predicts via ``predict``; ``task="classification"``
    trains on BCE logits and predicts via ``predict_proba``.
    """

    def __init__(
        self,
        task: Task,
        hidden_size: int | None = None,
        lr: float = 1e-2,
        epochs: int = 200,
        weight_decay: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.task = task
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(device)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "task": self.task,
            "hidden_size": self.hidden_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "device": str(self.device),
        }

    def set_params(self, **params: Any) -> "_TorchProbe":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _build_model(self, input_dim: int) -> nn.Module:
        # Same probe shape as EmbeddingProbeHead (model/heads.py).
        projection, classifier = build_probe_layers(input_dim, self.hidden_size, 1)
        return nn.Sequential(projection, classifier)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchProbe":
        torch.manual_seed(self.seed)
        self._model = self._build_model(X.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = _TORCH_PROBE_LOSS[self.task]
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_t).squeeze(-1), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self._model(X_t).squeeze(-1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            positive = torch.sigmoid(self._model(X_t).squeeze(-1)).cpu().numpy()
        return np.stack([1.0 - positive, positive], axis=1)


# probe_type="linear" maps onto the pipeline's classical sklearn_linear head
# (model/heads.py), which is Ridge/LogisticRegression under the hood.
_LINEAR_HEAD_TYPE: HeadType = "sklearn_linear"
_TASK_TYPE_BY_TASK: dict[Task, str] = {
    "regression": "sequence_regression",
    "classification": "sequence_classification",
}
_METRIC_NAME_BY_TASK: dict[Task, str] = {
    "regression": "spearmanr",
    "classification": "roc_auc",
}


def _make_probe(task: Task, probe_type: ProbeType, **hyperparameters: Any) -> Any:
    """Build the unscaled torch probe (``mlp``/``torch_linear``)."""
    if probe_type == "mlp":
        hyperparameters.setdefault("hidden_size", 512)
    else:
        hyperparameters.pop("hidden_size", None)
    return _TorchProbe(task, **hyperparameters)


class _ScaledEstimator:
    """A fitted probe plus the ``StandardScaler`` fit on its training embeddings."""

    def __init__(self, scaler: StandardScaler, estimator: Any) -> None:
        self.scaler = scaler
        self.estimator = estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self.scaler.transform(X))


def _score_candidate(
    task: Task, estimator: Any, val_X: np.ndarray, val_y: np.ndarray
) -> float:
    """Spearman (regression) or ROC-AUC (classification) on a validation split."""
    if task == "regression":
        return float(spearmanr(val_y, estimator.predict(val_X)).statistic)
    return float(roc_auc_score(val_y, estimator.predict_proba(val_X)[:, 1]))


def _fit_linear_probe(
    train_X: np.ndarray,
    train_y: np.ndarray,
    task: Task,
    hyperparameters: dict[str, Any],
    search_space: dict[str, list[Any]] | None,
    seed: int,
) -> Any:
    """Fit ``probe_type="linear"`` via the shared sklearn_linear head.

    ``build_estimator``/``sweep_classical_head`` already standardize features
    internally and filter C/alpha by task, so ``train_X`` is passed raw.
    """
    task_type = _TASK_TYPE_BY_TASK[task]
    if not search_space:
        estimator = build_estimator(
            _LINEAR_HEAD_TYPE, task_type, hyperparameters, seed=seed
        )
        estimator.fit(train_X, train_y)
        return estimator

    inner_train_X, inner_val_X, inner_train_y, inner_val_y = train_test_split(
        train_X,
        train_y,
        test_size=_TUNING_VAL_FRACTION,
        random_state=seed,
        stratify=train_y if task == "classification" else None,
    )
    _, best_hparams, _ = sweep_classical_head(
        head_type=_LINEAR_HEAD_TYPE,
        task_type=task_type,
        train_X=inner_train_X,
        train_y=inner_train_y,
        val_X=inner_val_X,
        val_y=inner_val_y,
        search_space=search_space,
        default_hyperparameters=hyperparameters,
        metric_name=_METRIC_NAME_BY_TASK[task],
        direction="maximize",
        search_pattern="grid",
        seed=seed,
    )
    estimator = build_estimator(_LINEAR_HEAD_TYPE, task_type, best_hparams, seed=seed)
    estimator.fit(train_X, train_y)
    return estimator


def _fit_probe(
    train_X: np.ndarray,
    train_y: np.ndarray,
    task: Task,
    probe_type: ProbeType,
    hyperparameters: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    seed: int = 1957723,
    device: torch.device | None = None,
) -> Any:
    """Fit *probe_type*, optionally sweeping ``search_space`` on an inner holdout.

    Candidates are scored on an inner validation split carved out of
    ``train_X``/``train_y``; the winning hyperparameters are then refit on
    the full ``train_X``/``train_y`` so the returned probe sees every row the
    caller made available.
    """
    hyperparameters = hyperparameters or {}
    if probe_type == "linear":
        return _fit_linear_probe(
            train_X, train_y, task, hyperparameters, search_space, seed
        )

    scaler = StandardScaler().fit(train_X)
    scaled_train_X = scaler.transform(train_X)
    if not search_space:
        if device is not None:
            hyperparameters = {**hyperparameters, "device": str(device)}
        estimator = _make_probe(task, probe_type, **hyperparameters)
        estimator.fit(scaled_train_X, train_y)
        return _ScaledEstimator(scaler, estimator)

    inner_train_X, inner_val_X, inner_train_y, inner_val_y = train_test_split(
        scaled_train_X,
        train_y,
        test_size=_TUNING_VAL_FRACTION,
        random_state=seed,
        stratify=train_y if task == "classification" else None,
    )
    best_params: dict[str, Any] = hyperparameters
    best_score = float("-inf")
    for candidate in ParameterGrid(search_space):
        params = {**hyperparameters, **candidate}
        if device is not None:
            params["device"] = str(device)
        estimator = _make_probe(task, probe_type, **params)
        estimator.fit(inner_train_X, inner_train_y)
        score = _score_candidate(task, estimator, inner_val_X, inner_val_y)
        if score > best_score:
            best_params, best_score = params, score

    if device is not None:
        best_params = {**best_params, "device": str(device)}
    best_estimator = _make_probe(task, probe_type, **best_params)
    best_estimator.fit(scaled_train_X, train_y)
    return _ScaledEstimator(scaler, best_estimator)


@torch.no_grad()
def pool_embeddings(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
    max_length: int | None = 1024,
) -> np.ndarray:
    """Mean-pool the trunk's last hidden state over valid (non-pad) tokens."""
    pooled = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoding = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]
        mean = pool_hidden_states(hidden, attention_mask, pooling="mean")
        pooled.append(mean.float().cpu().numpy())
    return np.concatenate(pooled, axis=0)


def resolve_folds(
    num_variants: int,
    official_folds: np.ndarray | None,
    strategy: SplitStrategy,
    n_splits: int = 5,
    seed: int = 1957723,
) -> np.ndarray:
    """Return one fold id per variant, from the official column or a fresh k-fold."""
    if strategy == "custom_kfold":
        folds = np.empty(num_variants, dtype=int)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (_, test_idx) in enumerate(kfold.split(np.arange(num_variants))):
            folds[test_idx] = fold_id
        return folds
    if official_folds is None:
        raise ValueError(
            f"Official split '{strategy}' requested but not present in the "
            "sourced assay; re-source with the dms_substitutions track or use "
            "split='custom_kfold'."
        )
    return np.asarray(official_folds)


def run_cv(
    embeddings: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    task: Task,
    probe_type: ProbeType = "linear",
    hyperparameters: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    seed: int = 1957723,
    device: torch.device | None = None,
) -> np.ndarray:
    """Out-of-fold predictions: fit on every other fold, predict the held-out one.

    Rows with a NaN target are excluded from training (they cannot inform a
    fit) but are still scored if held out. A fold is skipped -- leaving NaN
    predictions for its rows -- if it has no valid training rows, or if
    ``task='classification'`` and its training rows contain only one class
    (scikit-learn classifiers otherwise raise on the fit).
    """
    predictions = np.full(len(targets), np.nan)
    valid_target = ~np.isnan(targets)
    for fold_id in np.unique(folds):
        train_mask = (folds != fold_id) & valid_target
        test_mask = folds == fold_id
        if not train_mask.any():
            logger.warning("Skipping fold %s: no rows with a valid target", fold_id)
            continue
        if task == "classification" and len(np.unique(targets[train_mask])) < 2:
            logger.warning(
                "Skipping fold %s: training rows contain only one class", fold_id
            )
            continue
        estimator = _fit_probe(
            embeddings[train_mask],
            targets[train_mask],
            task,
            probe_type,
            hyperparameters=hyperparameters,
            search_space=search_space,
            seed=seed,
            device=device,
        )
        if task == "regression":
            predictions[test_mask] = estimator.predict(embeddings[test_mask])
        else:
            predictions[test_mask] = estimator.predict_proba(embeddings[test_mask])[
                :, 1
            ]
    return predictions


def run_seeded_split(
    embeddings: np.ndarray,
    targets: np.ndarray,
    split: np.ndarray,
    task: Task,
    probe_type: ProbeType = "linear",
    hyperparameters: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    seed: int = 1957723,
    device: torch.device | None = None,
) -> np.ndarray:
    """Fit on persisted train rows and return predictions for persisted test rows."""
    train_mask = (split == "train") & ~np.isnan(targets)
    test_mask = split == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Seeded split must contain both train and test rows")
    if task == "classification" and len(np.unique(targets[train_mask])) < 2:
        raise ValueError(
            "Seeded split's train rows contain only one class; cannot fit a "
            "classification probe"
        )
    estimator = _fit_probe(
        embeddings[train_mask],
        targets[train_mask],
        task,
        probe_type,
        hyperparameters=hyperparameters,
        search_space=search_space,
        seed=seed,
        device=device,
    )
    predictions = np.full(len(targets), np.nan)
    if task == "regression":
        predictions[test_mask] = estimator.predict(embeddings[test_mask])
    else:
        predictions[test_mask] = estimator.predict_proba(embeddings[test_mask])[:, 1]
    return predictions
