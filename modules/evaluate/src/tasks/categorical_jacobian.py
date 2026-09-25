"""Zero-shot contact prediction task backed by categorical Jacobian maps."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from modules.evaluate.src.model.categorical_jacobian import (
    DEFAULT_MAX_JACOBIAN_BYTES,
    build_categorical_jacobian_model,
)
from modules.evaluate.src.tasks.contact_prediction import ContactPredictionTask
from modules.evaluate.src.tasks.registry import register_task


@register_task
class CategoricalJacobianTask(ContactPredictionTask):
    """Contact prediction task that consumes Jacobian-derived pairwise scores."""

    name = "categorical_jacobian"
    MUTANT_BATCH_SIZE: ClassVar[int] = 256
    MAX_JACOBIAN_BYTES: ClassVar[int] = DEFAULT_MAX_JACOBIAN_BYTES

    def model_kwargs(self) -> dict[str, Any]:
        return {
            "mutant_batch_size": self.MUTANT_BATCH_SIZE,
            "max_jacobian_bytes": self.MAX_JACOBIAN_BYTES,
        }

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del num_labels
        return build_categorical_jacobian_model(
            model_id,
            **{**self.model_kwargs(), **(model_kwargs or {})},
        )

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return logits
