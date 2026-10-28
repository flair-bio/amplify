"""Compatibility shim; see ``modules.evaluate.scripts.dataset_sourcing.shared.preprocess``."""

from modules.evaluate.scripts.dataset_sourcing.shared.preprocess import (
    create_split_subsets,
    random_partition_indices,
)

__all__ = ["create_split_subsets", "random_partition_indices"]
