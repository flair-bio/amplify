"""Chance-level and substitution-matrix reference scorers for ProteinGym
zero-shot, run alongside the model for context.

``blosum62`` is a simple single-substitution-matrix score, not ProteinGym's
official "Site-Independent" baseline (which is MSA-derived) -- it is a
weaker, MSA-free reference, documented as such rather than presented as
matching the leaderboard's baseline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from modules.evaluate.src.proteingym.masked_marginal import parse_mutant

# Standard NCBI BLOSUM62 substitution matrix.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_INDEX = {aa: i for i, aa in enumerate(_BLOSUM62_ORDER)}
_BLOSUM62 = np.array(
    [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
    ],
    dtype=np.float64,
)


def score_assay_random(num_variants: int, seed: int = 0) -> np.ndarray:
    """Random scores: a chance-level sanity check (expected Spearman/AUC ~= 0)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(num_variants)


def score_assay_blosum62(mutants: Sequence[str]) -> np.ndarray:
    """Sum of BLOSUM62(wt_aa, mut_aa) substitution scores per (possibly multi-)mutant.

    Substitutions only -- indel assays have no ``mutant`` code to parse and
    should not be scored with this baseline. Mutants referencing an amino
    acid outside the standard 20 (e.g. ``X``, ``B``, ``Z``) score NaN rather
    than raising, consistent with other unscoreable-variant handling.
    """
    scores = np.zeros(len(mutants), dtype=np.float64)
    invalid = np.zeros(len(mutants), dtype=bool)
    mutant_idx, wt_idx, mut_idx = [], [], []
    for i, mutant in enumerate(mutants):
        for wt_aa, _, mut_aa in parse_mutant(mutant):
            if wt_aa not in _BLOSUM62_INDEX or mut_aa not in _BLOSUM62_INDEX:
                invalid[i] = True
                break
            mutant_idx.append(i)
            wt_idx.append(_BLOSUM62_INDEX[wt_aa])
            mut_idx.append(_BLOSUM62_INDEX[mut_aa])
    if mutant_idx:
        # Vectorized matrix lookup + per-mutant sum instead of accumulating in Python.
        np.add.at(scores, mutant_idx, _BLOSUM62[wt_idx, mut_idx])
    scores[invalid] = np.nan
    return scores
