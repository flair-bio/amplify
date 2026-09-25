"""Token-level classification: one class label per residue."""

from __future__ import annotations

import itertools
from typing import Any, ClassVar

import numpy as np
import torch
from transformers import AutoModelForTokenClassification

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task

# Loss/metric ignore index used for special-token and padding positions.
IGNORE_INDEX = -100


@register_task
class TokenClassificationTask(TaskHandler):
    """One class label per residue.

    Per-example predictions/labels are variable-length sequences: labels are
    aligned to the tokenized span during tokenization, padded to the batch
    width at collation, and stripped back to valid positions at prediction
    time. ``mcc`` complements the accuracy/f1/precision/recall metrics for
    imbalanced token labels.
    """

    name = "token_classification"
    auto_model_class = AutoModelForTokenClassification
    is_ragged = True
    aligns_labels = True
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"per_token"})

    metrics: ClassVar[dict[str, MetricFn]] = {
        "accuracy": metrics.accuracy,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "mcc": metrics.mcc,
    }
    default_metric_names = ("f1", "accuracy", "precision", "recall", "mcc")

    def validate_num_labels(
        self,
        dataset: Any,
        label_column: str,
        num_labels: int,
        dataset_name: str,
    ) -> None:
        self._validate_categorical_num_labels(
            dataset, label_column, num_labels, dataset_name
        )

    def align_labels(
        self,
        label: Any,
        *,
        num_residues: int,
        start: int,
        kept: int,
        offset: int,
        input_len: int,
        example_index: int,
    ) -> Any:
        if len(label) != num_residues:
            raise ValueError(
                "Token-classification labels must contain one label per "
                f"tokenized residue; got {len(label)} labels and "
                f"{num_residues} residues at dataset index {example_index}."
            )
        cropped = label[start : start + kept]
        # Reserve special-token positions for the ignore index expected by the loss.
        aligned = [IGNORE_INDEX] * input_len
        aligned[offset : offset + len(cropped)] = cropped
        return aligned

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        # Labels were aligned during tokenization, so padding just extends the
        # ignore index through the batch width.
        padded = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=self.label_dtype)
        for i, label_seq in enumerate(labels):
            padded[i, : len(label_seq)] = torch.tensor(
                label_seq, dtype=self.label_dtype
            )
        return padded

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        rows: list[tuple[Any, Any]] = []
        for pred_row, label_row in zip(predictions_np, labels_np):
            valid = label_row != IGNORE_INDEX
            rows.append((pred_row[valid].tolist(), label_row[valid].tolist()))
        return rows

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        # Kept as one list per example so predictions can be keyed by id;
        # metrics need a single flat sequence.
        return list(itertools.chain.from_iterable(values))

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Scramble each sequence's labels among its own positions only.

        Permuting within a sequence destroys the token-level label association while
        preserving each sequence's own label distribution/composition.
        """
        scrambled: list[Any] = []
        for label_seq in labels:
            permuted_idx = rng.permutation(len(label_seq))
            scrambled.append([label_seq[j] for j in permuted_idx])
        return scrambled
