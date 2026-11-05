"""
Calibration KL, cross-fitted temperature scaling and cohort metrics.

Metric rows are ``[kl, kl_fit, temperature, pred_idx]`` predictions and
``[is_t, in_balanced, in_high_nt, fold, consensus_mask]`` labels (see
``tasks.calibration.CalibrationTask.score``). Cohorts are balanced T sites
(``t``), balanced NT sites (``nt``) and high-NT sites (``high_nt``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, xlogy

logger = logging.getLogger(__name__)

COHORTS = ("t", "nt", "high_nt")
NUM_STRATA = 18  # 9 conservation bins x {T, NT}
TEMPERATURE_BOUNDS = (0.25, 8.0)


def forward_kl(
    logits: np.ndarray, profiles: np.ndarray, temperature: Any
) -> np.ndarray:
    """``KL(profile || softmax(logits / T))`` per row, with ``0 log 0 = 0``."""
    scaled = logits / np.reshape(temperature, (-1, 1))
    return (
        xlogy(profiles, profiles).sum(1)
        - (profiles * scaled).sum(1)
        + logsumexp(scaled, axis=1)
    )


def fit_temperature(
    logits: np.ndarray, profiles: np.ndarray, strata: np.ndarray
) -> tuple[float, str]:
    """Minimize the equal-stratum mean KL over ``T`` in ``TEMPERATURE_BOUNDS``.

    The objective is convex in ``beta = 1 / T``, so the root of its monotonic
    derivative is the optimum. Returns ``T`` and the bound it hit, if any.
    """
    _, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    if len(counts) < NUM_STRATA:
        logger.warning(
            "Fitting temperature on %d of %d strata.", len(counts), NUM_STRATA
        )
    weights = 1.0 / (len(counts) * counts[inverse])
    target = (profiles * logits).sum(1)

    def derivative(beta: float) -> float:
        scaled = beta * logits
        probabilities = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        return float(weights @ ((probabilities * logits).sum(1) - target))

    lower, upper = TEMPERATURE_BOUNDS
    if derivative(1 / upper) >= 0:
        return upper, "upper"
    if derivative(1 / lower) <= 0:
        return lower, "lower"
    beta = brentq(derivative, 1 / upper, 1 / lower, xtol=1e-12, rtol=1e-12)
    return 1 / beta, "none"


def crossfit_temperatures(
    logits: np.ndarray,
    profiles: np.ndarray,
    strata: np.ndarray,
    folds: np.ndarray,
    fit_rows: np.ndarray,
) -> np.ndarray:
    """Per-row temperature fitted on ``fit_rows`` of the other folds (NaN if none)."""
    temperatures = np.full(len(folds), np.nan)
    for fold in np.unique(folds):
        train = fit_rows & (folds != fold)
        if not train.any():
            continue
        temperature, bound = fit_temperature(
            logits[train], profiles[train], strata[train]
        )
        temperatures[folds == fold] = temperature
        logger.info(
            "Fold %d: temperature %.4f (bound: %s, %d fitting sites).",
            fold,
            temperature,
            bound,
            train.sum(),
        )
    return temperatures


def _arrays(predictions: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(predictions, dtype=float).reshape(-1, 4),
        np.asarray(labels, dtype=float).reshape(-1, 5),
    )


def _cohort(labels: np.ndarray, cohort: str) -> np.ndarray:
    is_t, balanced = labels[:, 0] == 1, labels[:, 1] == 1
    return {
        "t": balanced & is_t,
        "nt": balanced & ~is_t,
        "high_nt": labels[:, 2] == 1,
    }[cohort]


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def cohort_kl(predictions: Any, labels: Any, average: str, *, cohort: str) -> float:
    preds, labels = _arrays(predictions, labels)
    return _mean_or_nan(preds[_cohort(labels, cohort), 0])


def cohort_score(
    predictions: Any, labels: Any, average: str, *, cohort: str, fitted: bool = False
) -> float:
    """``100 * exp(-mean KL)`` at ``T = 1`` or at the cross-fitted temperature."""
    preds, labels = _arrays(predictions, labels)
    return 100.0 * float(
        np.exp(-_mean_or_nan(preds[_cohort(labels, cohort), int(fitted)]))
    )


def cohort_accuracy(
    predictions: Any, labels: Any, average: str, *, cohort: str
) -> float:
    """Share of sites whose argmax is one of the (possibly tied) consensus residues."""
    preds, labels = _arrays(predictions, labels)
    rows = _cohort(labels, cohort)
    pred = preds[rows, 3].astype(np.int64)
    consensus_mask = labels[rows, 4].astype(np.int64)
    return _mean_or_nan(((consensus_mask >> pred) & 1).astype(bool))


def mean_temperature(predictions: Any, labels: Any, average: str) -> float:
    """Mean of the per-fold fitted temperatures."""
    preds, labels = _arrays(predictions, labels)
    folds = labels[:, 3]
    return _mean_or_nan(
        np.array([preds[folds == fold, 2][0] for fold in np.unique(folds)])
    )
