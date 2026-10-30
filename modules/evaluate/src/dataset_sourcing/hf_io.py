"""Compatibility shim; see ``modules.evaluate.scripts.dataset_sourcing.shared.hf_io``."""

from modules.evaluate.scripts.dataset_sourcing.shared.hf_io import (
    upload_dataset_artifact,
)

__all__ = ["upload_dataset_artifact"]
