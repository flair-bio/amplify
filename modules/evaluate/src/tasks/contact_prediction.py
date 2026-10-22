"""Pairwise residue-residue contact prediction (binary contact map).

Unlike ``token_classification``'s one-label-per-residue rows, labels here are
one-label-per-*pair*: a dense ``(S, S)`` map per example, built from a sparse
list of positive-contact ``[i, j]`` pairs (everything else between resolved
residues at a valid separation is an implicit negative). Only pairs with
tokenized separation ``>= MIN_SEP`` are scored;
closer pairs (trivially "in contact" by sequence adjacency) are masked out
via the ``-100`` ignore value, same as padding and leading/trailing
special-token positions (CLS/BOS, EOS/SEP), which are not residues and are
never scored as implicit negatives.

Per-example predictions and labels carry their metadata in explicit fields:
``{"scores": ..., "separations": ...}`` and ``{"length": ..., "values":
...}``. This lets precision-at-L rank each sequence's own candidates without
encoding sequence length as a dummy prediction. See ``metrics/contact_metrics.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Sequence

import numpy as np
import torch

from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.model.contact_model import (
    ContactPredictionModel,
    build_contact_model,
)
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task

# Loss/metric ignore index for padding and (unscored) short-range pairs.
IGNORE_INDEX = -100


@register_task
class ContactPredictionTask(TaskHandler):
    """Binary residue-residue contact prediction."""

    name = "contact_prediction"
    is_ragged = True
    pairwise = True
    aligns_labels = True
    label_dtype = torch.int8
    accepted_target_types = frozenset({"categorical"})
    accepted_label_formats = frozenset({"contact_pairs"})

    #: Tokenized-residue separation below which a pair is never scored (it's
    #: a sequence-adjacency contact, not a structural one).
    MIN_SEP: ClassVar[int] = 6
    #: Default rank of the bilinear pairwise projection (see
    #: model/contact_model.py); override per run via
    #: ``workspace.model_kwargs: {rank: ...}`` to sweep it.
    RANK: ClassVar[int] = 128
    #: Default BCE positive-class weight (contact maps are extremely
    #: imbalanced); override per run via ``workspace.model_kwargs: {pos_weight: ...}``.
    POS_WEIGHT: ClassVar[float] = 30.0

    metrics: ClassVar[dict[str, MetricFn]] = {
        **contact_metrics.PRECISION_METRICS,
        **contact_metrics.RANGE_AUC_METRICS,
    }
    default_metric_names = (
        *contact_metrics.PRECISION_METRICS,
        *contact_metrics.RANGE_AUC_METRICS,
    )

    def model_kwargs(self) -> dict[str, Any]:
        return {"rank": self.RANK, "pos_weight": self.POS_WEIGHT}

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return build_contact_model(
            model_id, **{**self.model_kwargs(), **(model_kwargs or {})}
        )

    def load_finetuned_model(self, model_path: str) -> Any:
        return ContactPredictionModel.from_pretrained(model_path)

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        # Not a class count (binary per-pair score); nothing derives from it.
        return 2

    # -- data ----------------------------------------------------------------

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
        """Crop+shift raw ``[i, j]`` contact pairs into tokenized coordinates.

        The aligned dict retains the cropped sequence length and resolved
        residue positions, excluding special tokens from candidate pairs.
        """
        end = start + kept
        if isinstance(label, dict):
            pairs = label["pairs"]
            source_valid_positions = label["valid_positions"] or []
        else:
            pairs = label
            source_valid_positions = range(num_residues)
        pairs = [
            [offset + (i - start), offset + (j - start)]
            for i, j in pairs
            if start <= i < end and start <= j < end
        ]
        valid_positions = [
            offset + (position - start)
            for position in source_valid_positions
            if start <= position < end
        ]
        return {
            "pairs": pairs,
            "valid_positions": valid_positions,
            "res_start": offset,
            "res_end": offset + kept,
            "length": kept,
        }

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        idx = torch.arange(seq_len)
        base_valid = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= self.MIN_SEP
        collated = torch.full(
            (batch_size, seq_len, seq_len), IGNORE_INDEX, dtype=self.label_dtype
        )
        for b, entry in enumerate(labels):
            res_start, res_end = entry["res_start"], entry["res_end"]
            resolved = torch.zeros(seq_len, dtype=torch.bool)
            resolved[entry.get("valid_positions", range(res_start, res_end))] = True
            valid = base_valid.clone()
            # Exclude padding *and* special-token positions (before res_start,
            # at/after res_end): only real residues can be scored, positive
            # or implicit-negative.
            valid[res_end:, :] = False
            valid[:, res_end:] = False
            valid[:res_start, :] = False
            valid[:, :res_start] = False
            valid &= resolved[:, None] & resolved[None, :]
            collated[b][valid] = 0
            for i, j in entry["pairs"]:
                if valid[i, j]:
                    collated[b, i, j] = 1
                    collated[b, j, i] = 1
        return collated

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        return logits.sigmoid()

    def split_batch_rows(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        *,
        residue_lengths: torch.Tensor | None = None,
    ) -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for row_index, (pred_mat, label_mat) in enumerate(zip(predictions, labels)):
            valid = label_mat != IGNORE_INDEX
            # Every scored pair appears twice (symmetric); count each once.
            upper = valid & torch.triu(torch.ones_like(valid), diagonal=1)
            i_idx, j_idx = upper.nonzero(as_tuple=True)
            scores = [float(score) for score in pred_mat[i_idx, j_idx].tolist()]
            seps = [int(sep) for sep in (j_idx - i_idx).tolist()]
            values = [int(value) for value in label_mat[i_idx, j_idx].tolist()]
            if residue_lengths is not None:
                length = int(residue_lengths[row_index])
            else:
                scored_cols = valid.any(dim=0).nonzero(as_tuple=True)[0]
                length = (
                    int(scored_cols.max() - scored_cols.min() + 1)
                    if scored_cols.numel()
                    else 0
                )
            pred_row = {"scores": scores, "separations": seps}
            label_row = {"length": length, "values": values}
            rows.append((pred_row, label_row))
        return rows

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        # Precision-at-L needs each sequence's own top-scored candidates and
        # length, so this task's metrics consume the per-sequence rows
        # directly instead of a pooled flat sequence.
        return values

    def compute_per_example_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> np.ndarray:
        """Compute contact scores once per protein for paired resampling.

        Rows follow the input order and columns follow metric_names. Each
        existing metric is a mean over proteins with defined contributions,
        so resampling these scores preserves its aggregation and NaN handling.
        """
        self.validate_metric_names(metric_names)
        return np.asarray(
            [
                [
                    self.metrics[name]([prediction], [label], average)
                    for name in metric_names
                ]
                for prediction, label in zip(predictions, labels)
            ],
            dtype=np.float64,
        ).reshape(len(predictions), len(metric_names))

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Permute each row's per-pair labels while retaining its length."""
        scrambled: list[Any] = []
        for row in labels:
            if not isinstance(row, dict):
                length, values = row[0], row[1:]
                permuted_idx = rng.permutation(len(values))
                scrambled.append([length] + [values[j] for j in permuted_idx])
                continue
            length, values = row["length"], row["values"]
            permuted_idx = rng.permutation(len(values))
            scrambled.append(
                {"length": length, "values": [values[j] for j in permuted_idx]}
            )
        return scrambled
