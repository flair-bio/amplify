"""Narrow compatibility export for the pre-refactor embedding helper path.

``modules.evaluate.src.utils.embed`` moved to
:mod:`modules.evaluate.src.predictors.embeddings` (shared feature extraction and
pooling) and :mod:`modules.evaluate.src.artifacts.feature_cache` (cache storage
and identity). Only the helper that the protected ProteinGym supervised runner
imports is re-exported here; new code must import from the new modules.
"""

from modules.evaluate.src.predictors.embeddings import pool_hidden_states

__all__ = ["pool_hidden_states"]
