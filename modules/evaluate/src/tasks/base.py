"""Flat task semantics: labels, decoding, metric units and scientific scoring."""

from __future__ import annotations

import warnings
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from modules.evaluate.src.metrics import metrics as metric_functions

if TYPE_CHECKING:
    from modules.evaluate.src.artifacts.results import (
        PredictOutput,
        ScoreOutput,
    )
    from modules.evaluate.src.config import (
        ScoreConfig,
    )
MetricFn = Callable[[Any, Any, str], float]


def _flatten_label_values(dataset: Any, label_column: str) -> set[int]:
    """Collect every distinct integer label value across all splits of
    *dataset*, flattening one level of nesting (e.g. per-token label lists)."""
    values: set[int] = set()
    for split in dataset.values():
        for label in split[label_column]:
            if isinstance(label, (list, tuple)):
                values.update((int(v) for v in label))
            else:
                values.add(int(label))
    return values


class FlatTask(ABC):
    """Base class for a downstream evaluation task type."""

    name: ClassVar[str]
    class_labels: tuple[int, ...] | None = None
    uses_predictor_identity: ClassVar[bool] = False
    supports_packing: ClassVar[bool] = True
    supports_layerwise: ClassVar[bool] = False
    fixed_protocol: ClassVar[bool] = False
    evaluation_modes: ClassVar[frozenset[str]] = frozenset(
        {"evaluate_as_is", "tune_probe", "finetune"}
    )
    supported_head_types: ClassVar[frozenset[str]] = frozenset(
        {"mlp", "torch_linear", "sklearn_linear", "knn", "random_forest", "xgboost"}
    )
    supported_controls: ClassVar[frozenset[str]] = frozenset(
        {
            "untrained",
            "random_trunk",
            "random_head",
            "random_both",
            "scrambled_labels",
            "scrambled_sequences",
        }
    )
    dataset_baselines: ClassVar[frozenset[str]] = frozenset()
    prediction_baselines: ClassVar[frozenset[str]] = frozenset()
    label_dtype: ClassVar[torch.dtype] = torch.long
    is_ragged: ClassVar[bool] = False
    pairwise: ClassVar[bool] = False
    aligns_labels: ClassVar[bool] = False
    prediction_progress_unit: ClassVar[str] = "examples"
    requires_source_labels: ClassVar[bool] = True
    tokenization_batch_size: ClassVar[int] = 1000
    accepted_target_types: ClassVar[frozenset[str]] = frozenset()
    accepted_label_formats: ClassVar[frozenset[str]] = frozenset()
    metrics: ClassVar[Mapping[str, MetricFn]] = {}
    metrics_requiring_probabilities: ClassVar[frozenset[str]] = frozenset()
    default_metric_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, *, abstract: bool = False, **kwargs: Any) -> None:
        """Warn at class-definition time, not at first use, if a concrete
        handler forgets the fields every task type must set."""
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        if not getattr(cls, "name", None):
            warnings.warn(f"{cls.__name__} must set a non-empty 'name'.", stacklevel=2)
        if not cls.metrics:
            warnings.warn(
                f"{cls.__name__} must define at least one metric.", stacklevel=2
            )
        if cls.requires_source_labels and (
            not (cls.accepted_target_types and cls.accepted_label_formats)
        ):
            warnings.warn(
                f"{cls.__name__} consumes source labels, so it must set 'accepted_target_types' and 'accepted_label_formats'.",
                stacklevel=2,
            )
        if not cls.default_metric_names:
            warnings.warn(
                f"{cls.__name__} must set 'default_metric_names'.", stacklevel=2
            )

    def validate_head(self, head_type: str) -> None:
        if head_type not in self.supported_head_types:
            raise ValueError(
                f"head_type='{head_type}' is not supported for task_type='{self.name}'."
            )

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        """Resolve the effective label count from the workspace config.

        Supplied explicitly in the workspace configuration rather than
        inferred from the dataset's features.
        """
        if num_labels is None:
            raise ValueError(
                f"workspace.num_labels must be set for task_type='{self.name}' (dataset '{dataset_name}')."
            )
        if num_labels <= 1:
            raise ValueError(
                f"workspace.num_labels must be > 1 for task_type='{self.name}' (dataset '{dataset_name}'), got {num_labels}."
            )
        return num_labels

    def validate_num_labels(
        self, dataset: Any, label_column: str, num_labels: int, dataset_name: str
    ) -> None:
        """Cross-check the resolved ``workspace.num_labels`` against label
        values actually present in the loaded (pre-tokenization) dataset.

        Config-level validation only sees the pydantic ``num_labels`` value,
        never real label data, so a mismatch (e.g. ``num_labels=2`` against a
        10-class dataset) would otherwise only surface as a cryptic
        index-out-of-bounds/CUDA assert deep into training. No-op by default;
        only tasks whose classification head size is directly derived from
        ``workspace.num_labels`` (categorical scalar/per-token tasks)
        override this via :meth:`_validate_categorical_num_labels`.
        """
        return

    def _validate_categorical_num_labels(
        self,
        dataset: Any,
        label_column: str,
        num_labels: int,
        dataset_name: str,
        ignore_index: int | None = None,
    ) -> None:
        """Shared implementation for categorical tasks: raise if any label
        value in *dataset* falls outside ``range(num_labels)``."""
        values = _flatten_label_values(dataset, label_column)
        if ignore_index is not None:
            values.discard(ignore_index)
        if not values:
            return
        max_label = max(values)
        if max_label >= num_labels or min(values) < 0:
            raise ValueError(
                f"task_type='{self.name}' dataset '{dataset_name}' contains label value(s) outside the valid range for workspace.num_labels={num_labels} (valid indices are 0..{num_labels - 1}); found {len(values)} distinct label value(s) with max={max_label}. Set workspace.num_labels to at least {max_label + 1} to match the dataset."
            )

    def validate_dataset_config(
        self, target_type: str, label_format: str, dataset_name: str
    ) -> None:
        if not self.requires_source_labels:
            return
        if target_type not in self.accepted_target_types:
            raise ValueError(
                f"task_type='{self.name}' is incompatible with dataset '{dataset_name}': expected target_type in {sorted(self.accepted_target_types)}, got '{target_type}'."
            )
        if label_format not in self.accepted_label_formats:
            raise ValueError(
                f"task_type='{self.name}' is incompatible with dataset '{dataset_name}': expected label_format in {sorted(self.accepted_label_formats)}, got '{label_format}'."
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
        """Rewrite one example's label for the tokenized, cropped residue span.

        Only called when :attr:`aligns_labels` is set. ``start``/``kept`` are the
        crop window over the untokenized residues, ``offset`` is the number of
        leading special tokens, and ``input_len`` is the final ``input_ids``
        length.
        """
        return label

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        """Stack one batch's labels into the tensor the model head expects."""
        return torch.tensor(labels, dtype=self.label_dtype)

    def expand_tokenized_example(
        self,
        *,
        input_ids: list[int],
        attention_mask: list[int],
        label: Any,
        tokenizer: Any,
        res_start: int,
        res_end: int,
        example_id: Any,
        example_index: int,
    ) -> list[dict[str, Any]]:
        """Return one or more tokenized rows for a source example."""
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}
        ]

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reduce raw logits to predictions.

        ``labels`` is available for tasks whose prediction extraction depends
        on the batch labels, such as pseudo-perplexity scoring.
        """
        return logits.argmax(dim=-1)

    def predicted_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Per-example probabilities for probability-based metrics (e.g.
        ``roc_auc``). Defaults to softmax across classes (single-label);
        multilabel tasks override with an independent per-class sigmoid."""
        return logits.softmax(dim=-1)

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        """Split a gathered batch into aligned per-example (pred, label) pairs."""
        return list(zip(predictions.cpu().tolist(), labels.cpu().tolist()))

    def split_batch_rows_with_context(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        *,
        batch_metadata: Mapping[str, Any] | None = None,
    ) -> list[tuple[Any, Any]]:
        """Split rows while accepting optional task-specific batch metadata."""
        return self.split_batch_rows(predictions, labels)

    def bootstrap_group_id(self, row_id: Any) -> Any:
        """Map a row ID to the unit the bootstrap resamples as a whole."""
        return row_id

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        """Flatten per-example values into the flat sequence metrics expect."""
        return values

    def default_metrics(self) -> list[str]:
        return list(self.default_metric_names)

    def validate_metric_names(self, metric_names: Sequence[str]) -> None:
        unknown = sorted(set(metric_names) - set(self.metrics))
        if unknown:
            raise KeyError(
                f"Unknown metric(s) {unknown} for task_type='{self.name}'. Available metrics: {sorted(self.metrics)}."
            )

    def compute_per_example_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> np.ndarray | None:
        """Optionally compute scores that can be averaged over examples.

        Rows must follow the input order and columns must follow metric_names.
        NaN contributions are excluded independently for each metric. Return
        None when metrics require computation over the combined sample.
        """
        return None

    def validate_metric_inputs(self, predictions: Sequence, labels: Sequence) -> None:
        """Reject labels outside the configured vocabulary rather than omit them."""
        if self.class_labels is None:
            return
        pred_array, label_array = (np.asarray(predictions), np.asarray(labels))
        if pred_array.shape != label_array.shape:
            raise ValueError("Classification prediction and label shapes must match.")
        if self.name == "sequence_multilabel_classification":
            if label_array.ndim != 2 or label_array.shape[1] != len(self.class_labels):
                raise ValueError(
                    "Multilabel metric inputs must have one column per configured label."
                )
            if (
                not np.isin(pred_array, (0, 1)).all()
                or not np.isin(label_array, (0, 1)).all()
            ):
                raise ValueError("Multilabel metric inputs must contain binary values.")
        elif (
            pred_array.ndim != 1
            or not np.isin(pred_array, self.class_labels).all()
            or (not np.isin(label_array, self.class_labels).all())
        ):
            raise ValueError(
                "Classification metric inputs contain labels outside the configured class vocabulary; token ignore-index positions must be removed first."
            )

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Permute labels to build a chance-level control.

        Predictions are unchanged, but their association with the true labels
        is destroyed, so the resulting metric approximates what's attainable by
        chance given the label distribution.
        """
        permuted_idx = rng.permutation(len(labels))
        return [labels[j] for j in permuted_idx]

    def compute_grouped_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        group_ids: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
    ) -> dict[str, float]:
        """Compute validation metrics over task-defined groups."""
        raise ValueError(f"Grouped metrics are not supported for task '{self.name}'.")

    def prepare_scoring(
        self, predictions: PredictOutput, config: ScoreConfig
    ) -> tuple[FlatTask, PredictOutput]:
        """Resolve scoring units without changing persisted prediction rows."""
        config.validate_task(self.name)
        return (self, predictions)

    def score(
        self, predictions: PredictOutput, config: ScoreConfig, *, seed: int = 710019
    ) -> ScoreOutput:
        from modules.evaluate.src.metrics.scoring import score_flat

        return score_flat(self, predictions, config, seed=seed)

    def compute_metrics(
        self,
        predictions: Sequence,
        labels: Sequence,
        metric_names: Sequence[str],
        average: str = "weighted",
        probabilities: Sequence | None = None,
    ) -> dict[str, float]:
        """Compute the requested metrics for a completed evaluation pass.

        ``predictions``/``labels`` must already be flattened via
        :meth:`flatten_for_metrics` and stripped of any ignore-index positions.
        """
        if not len(predictions):
            raise ValueError("Cannot compute metrics from an empty predictions list.")
        self.validate_metric_names(metric_names)
        self.validate_metric_inputs(predictions, labels)
        scores: dict[str, float] = {}
        for name in metric_names:
            if name in self.metrics_requiring_probabilities:
                if probabilities is None:
                    raise ValueError(
                        f"Metric '{name}' requires per-class probabilities, but none were provided."
                    )
                scores[name] = float(self.metrics[name](probabilities, labels, average))
            elif (
                self.class_labels is not None
                and name in {"f1", "precision", "recall"}
                and (self.metrics[name] is getattr(metric_functions, name))
            ):
                scores[name] = float(
                    getattr(metric_functions, name)(
                        predictions, labels, average, class_labels=self.class_labels
                    )
                )
            else:
                scores[name] = float(self.metrics[name](predictions, labels, average))
        return scores
