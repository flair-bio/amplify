"""Compatibility shim; see ``modules.evaluate.scripts.dataset_sourcing.shared.outputs``."""

from modules.evaluate.scripts.dataset_sourcing.shared.outputs import (
    generated_utc_date,
    write_dataset_stats,
)

__all__ = ["generated_utc_date", "write_dataset_stats"]
