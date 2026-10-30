"""Reference-corpus priors for pseudo-perplexity and contact baselines (protocol v1).

Both estimators consume training data only. Callers pass the training split of
a declared reference corpus; test sequences and test pairs are never accepted
here.

* Amino-acid frequency (pseudo-perplexity): additive (Laplace) smoothing over
  the 20 canonical residues with pseudocount 1, so no canonical residue has
  zero probability. Non-canonical characters are excluded from the counts.
* Separation prior (contact prediction): P(contact | |i-j|) over
  original-coordinate pairs with separation at least ``MIN_SEPARATION`` (6,
  the contact-metric convention), with pseudocounts (1 contact, 2 pairs). A
  separation with no training pairs uses the pooled training contact rate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from modules.evaluate.src.predictors.composition import CANONICAL_AMINO_ACIDS

AMINO_ACID_FREQUENCY_PROTOCOL: dict[str, Any] = {
    "version": 1,
    "reference": "training split of a declared reference corpus only",
    "alphabet": CANONICAL_AMINO_ACIDS,
    "smoothing": "additive (Laplace) pseudocount 1 per canonical residue",
    "unknown_residues": "excluded from counts; never assigned probability",
}

MIN_SEPARATION = 6
SEPARATION_PRIOR_PROTOCOL: dict[str, Any] = {
    "version": 1,
    "reference": "training split pairs only",
    "min_separation": MIN_SEPARATION,
    "smoothing": "P = (contacts + 1) / (pairs + 2) per separation",
    "unseen_separation": "pooled training contact rate, smoothed the same way",
}


def fit_amino_acid_frequencies(reference_sequences: Iterable[str]) -> np.ndarray:
    """Return smoothed probabilities over ``CANONICAL_AMINO_ACIDS``."""
    index = {residue: i for i, residue in enumerate(CANONICAL_AMINO_ACIDS)}
    counts = np.ones(len(CANONICAL_AMINO_ACIDS), dtype=np.float64)
    for sequence in reference_sequences:
        for residue in sequence.upper():
            position = index.get(residue)
            if position is not None:
                counts[position] += 1.0
    return counts / counts.sum()


def amino_acid_log_probability(frequencies: np.ndarray, residue: str) -> float:
    """Natural-log probability of one canonical residue."""
    position = CANONICAL_AMINO_ACIDS.find(residue.upper()) if len(residue) == 1 else -1
    if position < 0:
        raise ValueError(
            f"No amino-acid frequency for non-canonical residue {residue!r}."
        )
    return float(np.log(frequencies[position]))


@dataclass(frozen=True)
class SeparationPrior:
    """Smoothed contact probabilities by separation plus the pooled rate."""

    by_separation: Mapping[int, float]
    pooled: float

    def probability(self, separation: int) -> float:
        """Probability for one separation, using the pooled rate when unseen."""
        if separation < MIN_SEPARATION:
            raise ValueError(f"Separation {separation} is outside the scored range.")
        return float(self.by_separation.get(separation, self.pooled))


def fit_separation_prior(pairs: Iterable[tuple[int, bool]]) -> SeparationPrior:
    """Fit contact probabilities from training ``(separation, is_contact)`` pairs.

    Pairs with separation below ``MIN_SEPARATION`` are ignored, matching the
    scored range.
    """
    contacts: Counter[int] = Counter()
    totals: Counter[int] = Counter()
    for separation, is_contact in pairs:
        if separation < MIN_SEPARATION:
            continue
        totals[separation] += 1
        if is_contact:
            contacts[separation] += 1
    by_separation = {
        separation: (contacts[separation] + 1.0) / (totals[separation] + 2.0)
        for separation in sorted(totals)
    }
    pooled = (sum(contacts.values()) + 1.0) / (sum(totals.values()) + 2.0)
    return SeparationPrior(by_separation=by_separation, pooled=pooled)
