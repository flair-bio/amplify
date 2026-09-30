"""Tests for modules.evaluate.src.dataset.collator.EvaluationCollator.

Uses the real ``ProteinTokenizer`` (see tests/evaluate/conftest.py) so
batches are built from genuinely tokenized sequences (real BOS/EOS/pad
token ids), not mocks.
"""

from __future__ import annotations

import math

import torch

from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)

from .conftest import LONG_SEQUENCE, SHORT_SEQUENCE

_SEQUENCES = [SHORT_SEQUENCE, LONG_SEQUENCE, SHORT_SEQUENCE[:5]]


def _tokenize(tokenizer, sequence: str) -> dict[str, list[int]]:
    encoded = tokenizer(sequence, add_special_tokens=True)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def _seq_batch(tokenizer, targets, label_column="targets") -> list[dict]:
    batch = []
    for seq, target in zip(_SEQUENCES, targets):
        example = _tokenize(tokenizer, seq)
        example[label_column] = target
        batch.append(example)
    return batch


def _token_batch(tokenizer, label_column="targets") -> list[dict]:
    """Per-token labels already aligned with -100 at BOS/EOS, as tokenize_dataset produces."""
    batch = []
    for seq in _SEQUENCES:
        example = _tokenize(tokenizer, seq)
        n = len(example["input_ids"])
        # Real residue labels alternate 0/1; BOS (first) and EOS (last) are -100.
        labels = [-100] + [i % 2 for i in range(n - 2)] + [-100]
        example[label_column] = labels
        batch.append(example)
    return batch


# ---------------------------------------------------------------------------
# Padded mode: sequence_classification / sequence_regression
# ---------------------------------------------------------------------------


class TestPaddedSequenceTasks:
    def test_sequence_classification_shapes_and_padding(self, protein_tokenizer):
        tok = protein_tokenizer
        targets = [0, 1, 0]
        batch = _seq_batch(tok, targets)

        collator = EvaluationCollator(
            tok,
            "sequence_classification",
            EvaluationCollatorConfig(pad_to_multiple_of=8, label_column="targets"),
        )
        out = collator(batch)

        max_len = max(len(ex["input_ids"]) for ex in batch)
        expected_len = math.ceil(max_len / 8) * 8

        assert out["input_ids"].shape == (3, expected_len)
        assert out["attention_mask"].shape == (3, expected_len)
        assert out["labels"].dtype == torch.long
        assert out["labels"].tolist() == targets

        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            assert out["input_ids"][i, :n].tolist() == ex["input_ids"]
            assert torch.all(out["input_ids"][i, n:] == tok.pad_token_id)
            assert torch.all(out["attention_mask"][i, :n])
            assert not torch.any(out["attention_mask"][i, n:])

    def test_sequence_regression_labels_are_float(self, protein_tokenizer):
        tok = protein_tokenizer
        targets = [0.5, -1.25, 2.0]
        batch = _seq_batch(tok, targets)

        collator = EvaluationCollator(
            tok, "sequence_regression", EvaluationCollatorConfig(label_column="targets")
        )
        out = collator(batch)

        assert out["labels"].dtype == torch.float
        assert torch.allclose(out["labels"], torch.tensor(targets))

    def test_pad_to_multiple_of_none_uses_exact_max_length(self, protein_tokenizer):
        tok = protein_tokenizer
        batch = _seq_batch(tok, [0, 1, 0])

        collator = EvaluationCollator(
            tok,
            "sequence_classification",
            EvaluationCollatorConfig(pad_to_multiple_of=None, label_column="targets"),
        )
        out = collator(batch)

        max_len = max(len(ex["input_ids"]) for ex in batch)
        assert out["input_ids"].shape[1] == max_len


# ---------------------------------------------------------------------------
# Padded mode: token_classification
# ---------------------------------------------------------------------------


class TestPaddedTokenClassification:
    def test_labels_padded_with_ignore_index(self, protein_tokenizer):
        tok = protein_tokenizer
        batch = _token_batch(tok)

        collator = EvaluationCollator(
            tok,
            "token_classification",
            EvaluationCollatorConfig(pad_to_multiple_of=None, label_column="targets"),
        )
        out = collator(batch)

        seq_len = out["input_ids"].shape[1]
        assert out["labels"].shape == (len(batch), seq_len)
        assert out["labels"].dtype == torch.long

        for i, ex in enumerate(batch):
            n = len(ex["targets"])
            assert out["labels"][i, :n].tolist() == ex["targets"]
            # Beyond the real sequence, padded positions are also -100.
            assert torch.all(out["labels"][i, n:] == -100)
            # BOS/EOS positions are always -100.
            assert out["labels"][i, 0].item() == -100
            assert out["labels"][i, n - 1].item() == -100


# ---------------------------------------------------------------------------
# Attention mask representation
# ---------------------------------------------------------------------------


class TestAttentionMask:
    def test_mask_is_bool_and_no_position_ids(self, protein_tokenizer):
        """Models coerce the mask with .bool() and derive their own positions;
        a float mask silently inverts, and arange positions break ESM."""
        tok = protein_tokenizer
        batch = _seq_batch(tok, [0, 1, 0])
        out = EvaluationCollator(
            tok,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets"),
        )(batch)
        assert out["attention_mask"].dtype == torch.bool
        assert "position_ids" not in out


class TestPackedSequences:
    def test_packed_batch_preserves_boundaries_and_labels(self, protein_tokenizer):
        batch = _token_batch(protein_tokenizer)
        out = EvaluationCollator(
            protein_tokenizer,
            "token_classification",
            EvaluationCollatorConfig(packed=True, pad_to_multiple_of=8),
        )(batch)
        lengths = [len(example["input_ids"]) for example in batch]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        expected_size = math.ceil(offsets[-1] / 8) * 8

        assert out["input_ids"].shape == (1, expected_size)
        assert out["position_ids"].shape == (1, expected_size)
        assert out["cu_seqlens"].dtype == torch.int32
        assert out["cu_seqlens"].tolist() == offsets + [expected_size]
        assert out["max_seqlen"] == max(max(lengths), expected_size - offsets[-1])
        assert out["num_sequences"] == len(batch)
        assert "attention_mask" not in out
        for index, example in enumerate(batch):
            start, end = offsets[index : index + 2]
            assert out["input_ids"][0, start:end].tolist() == example["input_ids"]
            assert out["position_ids"][0, start:end].tolist() == list(
                range(lengths[index])
            )
            assert out["labels"][0, start:end].tolist() == example["targets"]
        assert out["labels"].shape == (1, expected_size)
        assert torch.all(out["labels"][0, offsets[-1] :] == -100)

    def test_sequence_labels_remain_per_example(self, protein_tokenizer):
        out = EvaluationCollator(
            protein_tokenizer,
            "sequence_regression",
            EvaluationCollatorConfig(packed=True),
        )(_seq_batch(protein_tokenizer, [0.5, -1.25, 2.0]))
        assert out["labels"].shape == (3,)
        assert out["labels"].dtype == torch.float
