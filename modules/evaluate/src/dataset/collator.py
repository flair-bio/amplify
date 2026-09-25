"""
Collate pre-tokenized examples into the tensors consumed by evaluation.

The workflow is deliberately simple: ``tokenize_dataset`` has already done the
expensive work, so this class only pads each batch to the local maximum length
and reshapes the masks/labels into the representation expected by the model
head.

Batches are padded to the batch's max length ``(B, S)``, matching
``AMPLIFYForSequenceClassification`` / ``AMPLIFYForTokenClassification``
(and equivalent HF Auto* heads), which accept ``input_ids``,
``attention_mask``, and task-appropriate ``labels``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.tasks import TaskType, get_task


def unpack_packed(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_sequences: int,
    pad_value: int | float = 0,
) -> torch.Tensor:
    offsets = cu_seqlens[: num_sequences + 1].tolist()
    return pad_sequence(
        [tensor[0, start:end] for start, end in zip(offsets[:-1], offsets[1:])],
        batch_first=True,
        padding_value=pad_value,
    )


def input_id_rows(batch: dict[str, Any], pad_token_id: int) -> torch.Tensor:
    if "cu_seqlens" not in batch:
        return batch["input_ids"]
    return unpack_packed(
        batch["input_ids"], batch["cu_seqlens"], batch["num_sequences"], pad_token_id
    )


class EvaluationCollatorConfig(BaseModel):
    """Config. for :class:`EvaluationCollator`."""

    model_config = ConfigDict(extra="forbid")

    pad_to_multiple_of: int | None = Field(
        8, description="Pad sequence length to a multiple of this value."
    )
    packed: bool = Field(
        False, description="Pack tokenized sequences for varlen attention."
    )
    label_column: str = Field(
        "targets", description="Dataset column containing the label(s)."
    )
    id_column: str = Field(
        "id",
        description=(
            "Dataset column containing a per-example identifier, passed "
            "through the batch (as a Python list, not a tensor) so "
            "predictions/labels can be keyed back to their source example. "
            "Omitted from the batch entirely when the dataset has no such "
            "column."
        ),
    )


class EvaluationCollator:
    """Batches pre-tokenized examples for a downstream evaluation task.

    Produces ``input_ids``, ``attention_mask``, and task-appropriate
    ``labels``; the label shape and dtype are owned by the task handler (see
    ``modules/evaluate/src/tasks/``).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        task_type: TaskType,
        config: EvaluationCollatorConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.task_type = task_type
        self.task = get_task(task_type)
        self.pad_to_multiple_of = config.pad_to_multiple_of
        self.packed = config.packed
        self.label_column = config.label_column
        self.id_column = config.id_column

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        output = (
            self._collate_packed(batch) if self.packed else self._collate_padded(batch)
        )
        if self.id_column in batch[0]:
            # Left as a plain Python list (not a tensor): callers must pop it
            # before feeding the batch to the model.
            output["id"] = [example[self.id_column] for example in batch]
        return output

    def _collate_packed(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [len(example["input_ids"]) for example in batch]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        total = offsets[-1]
        buf_size = total
        if self.pad_to_multiple_of is not None:
            buf_size = (
                math.ceil(total / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full((1, buf_size), pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((1, buf_size), dtype=torch.long)
        labels = [example[self.label_column] for example in batch]
        token_labels = isinstance(labels[0], (list, tuple))
        if token_labels:
            packed_labels = torch.full((1, buf_size), -100, dtype=self.task.label_dtype)
        else:
            packed_labels = self.task.collate_labels(labels, len(batch), 0)
        for index, example in enumerate(batch):
            start, end = offsets[index : index + 2]
            input_ids[0, start:end] = torch.as_tensor(
                example["input_ids"], dtype=torch.long
            )
            position_ids[0, start:end] = torch.arange(lengths[index])
            if token_labels:
                packed_labels[0, start:end] = torch.as_tensor(
                    labels[index], dtype=self.task.label_dtype
                )
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cu_seqlens": torch.tensor(offsets + [buf_size], dtype=torch.int32),
            "max_seqlen": max(max(lengths), buf_size - total),
            "num_sequences": len(batch),
            "labels": packed_labels,
        }

    def _collate_padded(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Padded ``(B, S)`` tensors with an ``attention_mask``."""
        input_ids_list = [example["input_ids"] for example in batch]
        attention_mask_list = [example["attention_mask"] for example in batch]
        labels = [example[self.label_column] for example in batch]

        batch_size = len(batch)
        seq_len = max(len(ids) for ids in input_ids_list)
        if self.pad_to_multiple_of is not None and seq_len % self.pad_to_multiple_of:
            seq_len = (
                math.ceil(seq_len / self.pad_to_multiple_of) * self.pad_to_multiple_of
            )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            # Some tokenizers (e.g. GPT-2-style) have no dedicated pad
            # token; fall back to eos_token_id (or 0 if that's also unset)
            # rather than passing None into torch.full(), which raises
            # TypeError.
            pad_token_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else 0
            )
        # Start from the tokenizer's pad id so the resulting batch is valid for
        # any HF/AMPLIFY-style head that expects standard padded input ids.
        input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
            input_ids[i, : len(ids)] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, : len(mask)] = torch.as_tensor(mask, dtype=torch.bool)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": self.task.collate_labels(labels, batch_size, seq_len),
        }

        return output
