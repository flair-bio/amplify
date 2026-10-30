"""Hyperparameter candidate generation and trial helpers for MLP search.

These functions take the search-space configuration explicitly instead of
reading them from a training step, so they can be used by any predictor.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from modules.evaluate.src.utils.hpo import sample_candidates

logger = logging.getLogger(__name__)


class SearchSpace(Protocol):
    """Read-only view of the MLP hyperparameter search space used for candidates.

    Properties (not attributes) keep the protocol covariant, so pydantic configs
    with narrower field types such as ``Literal`` or ``list[PositiveInt]`` satisfy it.
    """

    @property
    def learning_rate(self) -> Sequence[float]: ...

    @property
    def weight_decay(self) -> Sequence[float]: ...

    @property
    def max_epochs(self) -> Sequence[int]: ...

    @property
    def weight_decay_sampling(self) -> str: ...

    @property
    def hidden_size(self) -> Sequence[int] | None: ...

    @property
    def batch_size(self) -> Sequence[int] | None: ...

    @property
    def hidden_depth(self) -> Sequence[int] | None: ...


def improves(
    score: float, best: float | None, direction: Literal["maximize", "minimize"]
) -> bool:
    """NaN never improves, and any real score beats a missing or NaN best."""
    if math.isnan(score):
        return False
    if best is None or math.isnan(best):
        return True
    return score > best if direction == "maximize" else score < best


def _axis(values: Sequence[int] | None) -> list[int | None]:
    """Candidate values for one optional hyperparameter axis (``None`` = unset)."""
    return list(values) if values else [None]


def probe_search_starters(
    space: SearchSpace,
) -> list[tuple[int | None, int | None, int | None]]:
    """Initial width/batch/depth combinations covering each configured value."""
    if (
        space.hidden_size is None
        and space.batch_size is None
        and space.hidden_depth is None
    ):
        return []
    widths = _axis(space.hidden_size)
    batches = _axis(space.batch_size)
    depths = _axis(space.hidden_depth)
    return [
        (
            widths[index % len(widths)],
            batches[index % len(batches)],
            depths[index % len(depths)],
        )
        for index in range(max(len(widths), len(batches), len(depths)))
    ]


def generate_candidates(
    space: SearchSpace,
    search_pattern: Literal["bohb", "random", "grid"],
    n_trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Enumerate or sample the trial configurations for one search."""
    combos: list[tuple[float, float, int, int | None, int | None, int | None]] = list(
        itertools.product(
            space.learning_rate,
            space.weight_decay,
            space.max_epochs,
            _axis(space.hidden_size),
            _axis(space.batch_size),
            _axis(space.hidden_depth),
        )
    )

    if search_pattern == "grid":
        # Grid search must exhaustively cover the space in a deterministic order,
        # not a shuffled/truncated subsample of it.
        if len(combos) > n_trials:
            logger.warning(
                "Grid search space has %d combinations, which exceeds "
                "n_trials=%d; running the full grid instead of "
                "truncating it (grid search should not subsample).",
                len(combos),
                n_trials,
            )
        candidates = combos
    else:  # "random"
        if n_trials > len(combos):
            logger.warning(
                "n_trials=%d exceeds the number of unique combinations "
                "(%d); running all unique combinations once instead.",
                n_trials,
                len(combos),
            )
        starters_hp = probe_search_starters(space)
        if not starters_hp:
            candidates = sample_candidates(combos, n_trials, seed)
        else:
            shuffled = sample_candidates(combos, len(combos), seed)
            starters = [
                (
                    space.learning_rate[0],
                    space.weight_decay[0],
                    space.max_epochs[0],
                    width,
                    batch,
                    depth,
                )
                for width, batch, depth in starters_hp
            ]
            candidates = (
                starters + [item for item in shuffled if item not in starters]
            )[:n_trials]

    return [
        {
            "learning_rate": lr,
            "weight_decay": wd,
            "max_epochs": max_epochs,
            "seed": seed,
            **({"hidden_size": hidden_size} if hidden_size is not None else {}),
            **({"batch_size": batch_size} if batch_size is not None else {}),
            **({"hidden_depth": hidden_depth} if hidden_depth is not None else {}),
        }
        for lr, wd, max_epochs, hidden_size, batch_size, hidden_depth in candidates
    ]


def suggest_weight_decay(trial: Any, space: SearchSpace) -> float:
    """Sample weight decay on the log scale or from the categorical list."""
    if space.weight_decay_sampling == "log":
        return trial.suggest_float(
            "weight_decay",
            min(space.weight_decay),
            max(space.weight_decay),
            log=True,
        )
    return trial.suggest_categorical("weight_decay", space.weight_decay)
