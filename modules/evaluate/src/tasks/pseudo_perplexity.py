"""Pseudo-perplexity for masked language models."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from transformers import AutoModelForMaskedLM

from modules.evaluate.src.metrics import metrics
from modules.evaluate.src.tasks.base import MetricFn, TaskHandler
from modules.evaluate.src.tasks.registry import register_task

IGNORE_INDEX = -100


@register_task
class PseudoPerplexityTask(TaskHandler):
    """Score each residue by masking it and reading its MLM log-probability."""

    name = "pseudo_perplexity"
    auto_model_class = AutoModelForMaskedLM
    requires_source_labels = False

    metrics: ClassVar[dict[str, MetricFn]] = {
        "pseudo_perplexity": metrics.pseudo_perplexity,
        "mean_log_prob": metrics.mean_log_prob,
    }
    default_metric_names = ("pseudo_perplexity",)

    def build_model(
        self,
        model_id: str,
        num_labels: int,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return self.auto_model_class.from_pretrained(
            model_id,
            trust_remote_code=True,
            **{**self.model_kwargs(), **(model_kwargs or {})},
        )

    def resolve_num_labels(self, num_labels: int | None, dataset_name: str) -> int:
        return 1

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
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("pseudo_perplexity requires a tokenizer mask token.")

        rows: list[dict[str, Any]] = []
        for pos in range(res_start, res_end):
            masked_ids = list(input_ids)
            labels = [IGNORE_INDEX] * len(input_ids)
            labels[pos] = input_ids[pos]
            masked_ids[pos] = mask_token_id
            rows.append(
                {
                    "input_ids": masked_ids,
                    "attention_mask": attention_mask,
                    "label": labels,
                    "id": example_id,
                }
            )
        return rows

    def collate_labels(
        self, labels: list[Any], batch_size: int, seq_len: int
    ) -> torch.Tensor:
        padded = torch.full((batch_size, seq_len), IGNORE_INDEX, dtype=self.label_dtype)
        for i, label_seq in enumerate(labels):
            padded[i, : len(label_seq)] = torch.tensor(
                label_seq, dtype=self.label_dtype
            )
        return padded

    def extract_predictions(
        self, logits: torch.Tensor, *, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError(
                "pseudo_perplexity requires labels to score masked tokens."
            )

        scored = labels != IGNORE_INDEX
        target_ids = labels[scored]
        log_probs = logits.log_softmax(dim=-1)[scored]
        return log_probs[
            torch.arange(target_ids.numel(), device=logits.device), target_ids
        ]

    def split_batch_rows(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> list[tuple[Any, Any]]:
        rows: list[tuple[Any, Any]] = []
        for pred, label_row in zip(predictions.cpu().tolist(), labels.cpu().tolist()):
            target = next(value for value in label_row if value != IGNORE_INDEX)
            rows.append((float(pred), int(target)))
        return rows
