"""Configuration for the scoring stage (metrics, bootstrap, long-format output)."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Minimum n_samples below which bootstrap confidence intervals and
# trained-vs-control p-values are too coarse/unstable to support a claim of
# statistical significance: with n resamples, the smallest achievable
# nonzero p-value is 1/n, and percentile-based CIs need enough resamples to
# actually populate their tails rather than interpolating from a handful of
# points. 1000 matches standard bootstrap guidance (Efron & Tibshirani).
_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES = 1000

# Permutation p-value resolution is 1/(n+1) regardless of bootstrap n_samples,
# and Benjamini-Hochberg across controls raises the smallest achievable
# adjusted value further, so this needs the same order of magnitude as the CI.
_DEFAULT_N_PERMUTATIONS = 1000


class ScoreConfig(BaseModel):
    """Config. for the score step in evaluation."""

    model_config = ConfigDict(extra="forbid")

    metrics: list[str] | None = None
    metric_aggregation: Literal["micro", "macro", "weighted", "samples"] = "weighted"

    # Bootstrap confidence intervals / control comparisons are optional
    # on top of the (always-computed) point-estimate metrics above. Default
    # n_samples matches _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES rather than an
    # arbitrarily small number, so enabling bootstrap without also setting
    # n_samples doesn't silently produce CIs/p-values too coarse to trust.
    bootstrap_enabled: bool = False
    n_samples: int = Field(_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES, gt=0)
    n_permutations: int = Field(_DEFAULT_N_PERMUTATIONS, gt=0)
    output_filename: str = "bootstrap_results.parquet"
    confidence_interval: float = Field(0.95, gt=0.0, lt=1.0)

    # Tidy/long-format sidecar (see ``build_long_df``), written on every run
    # (with or without bootstrap) so downstream plotting/aggregation code has
    # a single, concatenable artifact per (model, dataset) run regardless of
    # which optional outputs above were enabled.
    long_filename: str = "scores_long.parquet"

    @model_validator(mode="after")
    def _warn_if_resamples_too_low_for_significance(self) -> ScoreConfig:
        if not self.bootstrap_enabled:
            return self
        if self.n_samples < _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES:
            logger.warning(
                "ScoreConfig.n_samples=%d is well below the %d resamples "
                "generally recommended for reliable bootstrap confidence "
                "intervals (percentile intervals are estimated from only %d "
                "points). Do not use these results to support a claim of "
                "statistical significance without increasing n_samples.",
                self.n_samples,
                _MIN_RECOMMENDED_BOOTSTRAP_SAMPLES,
                self.n_samples,
            )
        if self.n_permutations < _DEFAULT_N_PERMUTATIONS:
            logger.warning(
                "ScoreConfig.n_permutations=%d floors every trained-vs-control "
                "p-value at 1/(n_permutations+1)=%.4f before Benjamini-Hochberg "
                "correction across controls raises it further; the conventional "
                "0.05/0.01 thresholds may be unreachable.",
                self.n_permutations,
                1.0 / (self.n_permutations + 1),
            )
        return self
