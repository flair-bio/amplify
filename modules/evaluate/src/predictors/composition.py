"""Composition-feature baselines with fixed, documented estimator settings.

Protocol (version 1), chosen for conservative, deterministic baselines:

* Features: residue frequencies over the 20 canonical amino acids. Input is
  upper-cased; every non-canonical character (ambiguous codes such as X, B, Z,
  U, O, and gaps) is excluded from the denominator. A sequence with no
  canonical residues has an all-zero profile.
* Classification: multinomial logistic regression with L2 penalty
  (``C=1.0``, lbfgs, ``max_iter=1000``, no class weights), fit on the training
  split only. If the training split observes fewer than two classes the
  estimator is constant (the observed class, probability 1), which is the only
  defensible prediction from a single class.
* Regression: ridge regression (``alpha=1.0``) on the same features, fit on the
  training split only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

CANONICAL_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

COMPOSITION_PROTOCOL: dict[str, Any] = {
    "version": 1,
    "alphabet": CANONICAL_AMINO_ACIDS,
    "normalization": (
        "frequency over canonical residues; non-canonical characters excluded "
        "from the denominator; no canonical residues gives a zero profile"
    ),
    "classification": {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
    },
    "regression": {
        "estimator": "sklearn.linear_model.Ridge",
        "alpha": 1.0,
    },
}

_INDEX = {residue: index for index, residue in enumerate(CANONICAL_AMINO_ACIDS)}


class ConstantClassifier:
    """Predict one observed class when the training split has a single class."""

    def __init__(self, label: int) -> None:
        self.label = label


def composition_features(sequences: list[str]) -> np.ndarray:
    """Return an ``(n, 20)`` matrix of canonical residue frequencies."""
    features = np.zeros((len(sequences), len(CANONICAL_AMINO_ACIDS)), dtype=np.float64)
    for row, sequence in enumerate(sequences):
        counts = np.zeros(len(CANONICAL_AMINO_ACIDS), dtype=np.float64)
        for residue in sequence.upper():
            index = _INDEX.get(residue)
            if index is not None:
                counts[index] += 1.0
        total = counts.sum()
        if total > 0:
            features[row] = counts / total
    return features


def fit_composition_classifier(
    train_sequences: list[str], labels: list[int], *, num_labels: int
) -> LogisticRegression | ConstantClassifier:
    """Fit the regularized multinomial classifier on training data only."""
    observed = sorted(set(labels))
    if len(observed) < 2:
        return ConstantClassifier(observed[0])
    protocol = COMPOSITION_PROTOCOL["classification"]
    model = LogisticRegression(
        C=protocol["C"],
        solver=protocol["solver"],
        max_iter=protocol["max_iter"],
        class_weight=protocol["class_weight"],
    )
    model.fit(composition_features(train_sequences), np.asarray(labels))
    return model


def predict_composition_classifier(
    model: LogisticRegression | ConstantClassifier,
    sequences: list[str],
    *,
    num_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, probabilities)`` over all ``num_labels`` classes."""
    probabilities = np.zeros((len(sequences), num_labels), dtype=np.float64)
    if isinstance(model, ConstantClassifier):
        probabilities[:, model.label] = 1.0
    else:
        observed = model.predict_proba(composition_features(sequences))
        probabilities[:, model.classes_] = observed
    return probabilities.argmax(axis=1), probabilities


def fit_composition_regressor(
    train_sequences: list[str], targets: list[float]
) -> Ridge:
    """Fit ridge regression on composition features using training data only."""
    protocol = COMPOSITION_PROTOCOL["regression"]
    model = Ridge(alpha=protocol["alpha"])
    model.fit(composition_features(train_sequences), np.asarray(targets, dtype=float))
    return model


def predict_composition_regressor(model: Ridge, sequences: list[str]) -> np.ndarray:
    return np.asarray(model.predict(composition_features(sequences)), dtype=float)
