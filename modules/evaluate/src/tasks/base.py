"""
Task-type abstraction for the evaluation pipeline.

A :class:`TaskHandler` owns everything that varies between downstream
evaluation tasks -- model class, label alignment/collation, prediction
extraction, metrics, and control-condition label scrambling -- so the
prepare/tune/predict/score steps stay task-agnostic. Adding a task type is a
new module under ``modules/evaluate/src/tasks/`` decorated with
``@register_task``; no step code needs to change.

The base class defaults describe the most common shape: a single scalar label
per example, a Hugging Face ``Auto*`` wrapper around the trunk, and dense
(non-ragged) per-example predictions. Handlers override only what differs.
"""

from __future__ import annotations

import warnings
from abc import ABC
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np
import torch
from torch import nn

# Every metric callable takes the same (values, labels, average) signature so
# the base compute_metrics() can dispatch without knowing the metric. ``values``
# is normally the predictions, but is the per-class probability matrix for
# metrics listed in ``metrics_requiring_probabilities`` (e.g. roc_auc).
MetricFn = Callable[[Any, Any, str], float]


def _flatten_label_values(dataset: Any, label_column: str) -> set[int]:
    """Collect every distinct integer label value across all splits of
    *dataset*, flattening one level of nesting (e.g. per-token label lists)."""
    values: set[int] = set()
    for split in dataset.values():
        for label in split[label_column]:
            if isinstance(label, (list, tuple)):
                values.update(int(v) for v in label)
            else:
                values.add(int(label))
    return values


class TaskHandler(ABC):
    """Base class for a downstream evaluation task type."""

    #: Value of ``workspace.task_type`` that selects this handler.
    name: ClassVar[str]

    #: Hugging Face ``Auto*`` class used to wrap the trunk. Works with any Hub
    #: model repo rather than depending on a specific model module. Subclasses
    #: that need a non-HF head override :meth:`build_model` instead.
    auto_model_class: ClassVar[Any]

    #: dtype of the collated ``labels`` tensor.
    label_dtype: ClassVar[torch.dtype] = torch.long

    #: Whether per-example predictions/labels are variable-length sequences.
    #: Ragged columns need object arrays for bootstrap fancy indexing.
    is_ragged: ClassVar[bool] = False

    #: Whether predictions/labels are per-*pair* (e.g. contact maps) rather
    #: than per-token/per-sequence.
    pairwise: ClassVar[bool] = False

    #: Whether tokenization must rewrite the label column to stay aligned with
    #: the tokenized (and possibly cropped) residue span.
    aligns_labels: ClassVar[bool] = False

    #: Whether the raw dataset must provide the configured label column.
    requires_source_labels: ClassVar[bool] = True

    accepted_target_types: ClassVar[frozenset[str]] = frozenset()
    accepted_label_formats: ClassVar[frozenset[str]] = frozenset()

    #: Metric name -> callable. Defines the full set this task supports.
    metrics: ClassVar[Mapping[str, MetricFn]] = {}

    #: Metrics that receive per-class probabilities instead of predictions.
    metrics_requiring_probabilities: ClassVar[frozenset[str]] = frozenset()

    #: Metrics computed when ``score.metrics`` is not set. Defaults to every
    #: supported metric: ``score.metrics`` is an exceptional override, not the
    #: primary way users discover the metric set.
    default_metric_names: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Warn at class-definition time, not at first use, if a concrete
        handler forgets the fields every task type must set."""
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "name", None):
            warnings.warn(f"{cls.__name__} must set a non-empty 'name'.", stacklevel=2)
        if cls.build_model is TaskHandler.build_model and not getattr(
            cls, "auto_model_class", None
        ):
            warnings.warn(
                f"{cls.__name__} must set 'auto_model_class' or override build_model().",
                stacklevel=2,
            )
        if not cls.metrics:
            warnings.warn(
                f"{cls.__name__} must define at least one metric.", stacklevel=2
            )
        if cls.requires_source_labels and not (
            cls.accepted_target_types and cls.accepted_label_formats
        ):
            warnings.warn(
                f"{cls.__name__} consumes source labels, so it must set "
                "'accepted_target_types' and 'accepted_label_formats'.",
                stacklevel=2,
            )
        if not cls.default_metric_names:
            warnings.warn(
                f"{cls.__name__} must set 'default_metric_names'.", stacklevel=2
            )

    # -- model ---------------------------------------------------------------

    def model_kwargs(self) -> dict[str, Any]:
        """Extra ``from_pretrained`` kwargs beyond the common ones."""
        return {}

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Load a fresh (pretrained-trunk, freshly-initialized-head) model.

        ``model_kwargs`` (from ``workspace.model_kwargs``) overrides this
        task's own :meth:`model_kwargs` defaults, e.g. to sweep a
        task-specific hyperparameter from the YAML config.
        """
        return self.auto_model_class.from_pretrained(
            model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            trust_remote_code=True,
            **{**self.model_kwargs(), **(model_kwargs or {})},
        )

    def load_finetuned_model(self, model_path: str) -> nn.Module:
        """Reload a previously saved, fine-tuned model.

        The head/label configuration is read back from the saved
        ``config.json``, so it doesn't need to be re-derived here.
        """
        return self.auto_model_class.from_pretrained(model_path, trust_remote_code=True)

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        """Resolve the effective label count from the workspace config.

        Supplied explicitly in the workspace configuration rather than
        inferred from the dataset's features.
        """
        if num_labels is None:
            raise ValueError(
                f"workspace.num_labels must be set for task_type='{self.name}' "
                f"(dataset '{dataset_name}')."
            )
        if num_labels <= 1:
            raise ValueError(
                f"workspace.num_labels must be > 1 for task_type='{self.name}' "
                f"(dataset '{dataset_name}'), got {num_labels}."
            )
        return num_labels

    def validate_num_labels(
        self,
        dataset: Any,
        label_column: str,
        num_labels: int,
        dataset_name: str,
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
    ) -> None:
        """Shared implementation for categorical tasks: raise if any label
        value in *dataset* falls outside ``range(num_labels)``."""
        values = _flatten_label_values(dataset, label_column)
        if not values:
            return
        max_label = max(values)
        if max_label >= num_labels or min(values) < 0:
            raise ValueError(
                f"task_type='{self.name}' dataset '{dataset_name}' contains "
                f"label value(s) outside the valid range for "
                f"workspace.num_labels={num_labels} (valid indices are "
                f"0..{num_labels - 1}); found {len(values)} distinct label "
                f"value(s) with max={max_label}. Set workspace.num_labels to "
                f"at least {max_label + 1} to match the dataset."
            )

    # -- data ----------------------------------------------------------------

    def validate_dataset_config(
        self, target_type: str, label_format: str, dataset_name: str
    ) -> None:
        if not self.requires_source_labels:
            return
        if target_type not in self.accepted_target_types:
            raise ValueError(
                f"task_type='{self.name}' is incompatible with dataset "
                f"'{dataset_name}': expected target_type in "
                f"{sorted(self.accepted_target_types)}, got '{target_type}'."
            )
        if label_format not in self.accepted_label_formats:
            raise ValueError(
                f"task_type='{self.name}' is incompatible with dataset "
                f"'{dataset_name}': expected label_format in "
                f"{sorted(self.accepted_label_formats)}, got '{label_format}'."
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
    ) -> list[dict[str, Any]]:
        """Return one or more tokenized rows for a source example."""
        return [
            {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}
        ]

    # -- prediction ----------------------------------------------------------

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reduce raw logits to predictions.

        ``labels`` is available for tasks whose prediction extraction depends
        on the batch labels, such as pseudo-perplexity scoring.
        """
        return logits.argmax(dim=-1)

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        """Split a gathered batch into aligned per-example (pred, label) pairs."""
        return list(zip(predictions.cpu().tolist(), labels.cpu().tolist()))

    # -- metrics -------------------------------------------------------------

    def flatten_for_metrics(self, values: list[Any]) -> list[Any]:
        """Flatten per-example values into the flat sequence metrics expect."""
        return values

    def default_metrics(self) -> list[str]:
        return list(self.default_metric_names)

    def validate_metric_names(self, metric_names: Sequence[str]) -> None:
        unknown = sorted(set(metric_names) - set(self.metrics))
        if unknown:
            raise KeyError(
                f"Unknown metric(s) {unknown} for task_type='{self.name}'. "
                f"Available metrics: {sorted(self.metrics)}."
            )

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

        scores: dict[str, float] = {}
        for name in metric_names:
            if name in self.metrics_requiring_probabilities:
                if probabilities is None:
                    raise ValueError(
                        f"Metric '{name}' requires per-class probabilities, but "
                        "none were provided."
                    )
                scores[name] = float(self.metrics[name](probabilities, labels, average))
            else:
                scores[name] = float(self.metrics[name](predictions, labels, average))
        return scores

    # -- controls ------------------------------------------------------------

    def scramble_labels(self, labels: list[Any], rng: np.random.Generator) -> list[Any]:
        """Permute labels to build a chance-level control.

        Predictions are unchanged, but their association with the true labels
        is destroyed, so the resulting metric approximates what's attainable by
        chance given the label distribution.
        """
        permuted_idx = rng.permutation(len(labels))
        return [labels[j] for j in permuted_idx]
