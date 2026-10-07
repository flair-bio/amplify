"""Unit tests for the MLM DataCollator (padded and packed paths)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from modules.pretrain.src.dataset.collator import (
    CollatorConfig,
    DataCollator,
    get_collator,
)
from modules.pretrain.src.model.tokenizer import ProteinTokenizer, TokenizerConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VOCAB = [
    "<pad>", "<unk>", "<mask>", "<bos>", "<eos>",
    "|", "X", "B", "O", "U", "Z", "J",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
]

_SEQUENCES = ["MKTAYIAKQR", "MGSSHH", "LAGVSERTIDPKQ"]


@pytest.fixture()
def tokenizer() -> ProteinTokenizer:
    return ProteinTokenizer(
        TokenizerConfig(
            vocab=_VOCAB,
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
            bos_token="<bos>",
            eos_token="<eos>",
        )
    )


def _collator(tokenizer: ProteinTokenizer, **overrides) -> DataCollator:
    defaults = dict(max_length=64, random_truncate=False)
    defaults.update(overrides)
    return get_collator(tokenizer, CollatorConfig(**defaults))


def _samples(sequences=_SEQUENCES) -> list[dict]:
    return [{"sequence": s} for s in sequences]


def _true_lengths(tokenizer: ProteinTokenizer, sequences=_SEQUENCES) -> np.ndarray:
    encoded = tokenizer(list(sequences), truncation=True, max_length=64)
    return np.array([len(ids) for ids in encoded["input_ids"]])


# ---------------------------------------------------------------------------
# Padded path
# ---------------------------------------------------------------------------


class TestCollatePadded:
    def test_shapes_and_attention_mask_track_real_lengths(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        batch = _collator(tokenizer, mlm=False)(_samples())

        assert batch["input_ids"].shape == (len(_SEQUENCES), int(lengths.max()))
        assert batch["attention_mask"].dtype == torch.bool
        assert batch["attention_mask"].sum(dim=1).tolist() == lengths.tolist()

    def test_padding_region_is_pad_id_and_zero_position(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        batch = _collator(tokenizer, mlm=False)(_samples())

        for row, n in enumerate(lengths):
            assert (batch["input_ids"][row, n:] == tokenizer.pad_token_id).all()
            assert (batch["position_ids"][row, n:] == 0).all()
            assert batch["position_ids"][row, :n].tolist() == list(range(int(n)))

    def test_pad_to_multiple_of_rounds_sequence_dim_up(self, tokenizer):
        batch = _collator(tokenizer, mlm=False, pad_to_multiple_of=8)(_samples())
        seq_len = batch["input_ids"].shape[1]

        assert seq_len % 8 == 0
        assert seq_len >= int(_true_lengths(tokenizer).max())

    def test_mlm_disabled_yields_all_ignored_labels(self, tokenizer):
        batch = _collator(tokenizer, mlm=False)(_samples())
        assert (batch["labels"] == -100).all()


# ---------------------------------------------------------------------------
# Packed path
# ---------------------------------------------------------------------------


class TestCollatePacked:
    def test_flat_buffer_and_token_accounting(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        batch = _collator(tokenizer, mlm=False, packing=True)(_samples())

        assert batch["input_ids"].shape == (1, int(lengths.sum()))
        assert batch["num_sequences"] == len(_SEQUENCES)
        assert batch["num_tokens"] == int(lengths.sum())

    def test_cu_seqlens_delimits_each_sequence(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        # Padding is on so the trailing segment is non-empty: cu_seqlens must
        # close on the buffer size, not on the real token count.
        batch = _collator(tokenizer, mlm=False, packing=True, pad_to_multiple_of=64)(
            _samples()
        )
        cu = batch["cu_seqlens"]
        buf = batch["input_ids"].shape[1]

        # One extra entry past the sequences closes the trailing padding segment.
        assert cu.shape == (len(_SEQUENCES) + 2,)
        assert cu.dtype == torch.int32
        assert cu[0] == 0
        assert torch.equal(torch.diff(cu)[: len(_SEQUENCES)], torch.tensor(lengths, dtype=torch.int32))
        assert cu[len(_SEQUENCES)] == int(lengths.sum())
        assert cu[-1] == buf
        assert buf > int(lengths.sum())
        assert (torch.diff(cu) >= 0).all()

    def test_position_ids_restart_per_packed_sequence(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        batch = _collator(tokenizer, mlm=False, packing=True)(_samples())
        pos = batch["position_ids"][0]

        offset = 0
        for n in lengths:
            n = int(n)
            assert pos[offset : offset + n].tolist() == list(range(n))
            offset += n

    def test_padding_segment_is_masked_out_and_counted_in_max_seqlen(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        # Chosen so the trailing padding segment is longer than the longest real
        # sequence; otherwise max_seqlen would be correct by accident.
        batch = _collator(tokenizer, mlm=False, packing=True, pad_to_multiple_of=64)(
            _samples()
        )
        total = int(lengths.sum())
        buf = batch["input_ids"].shape[1]
        padding_segment = buf - total

        assert buf % 64 == 0
        assert padding_segment > int(lengths.max())
        assert (batch["input_ids"][0, total:] == tokenizer.pad_token_id).all()
        assert (batch["labels"][0, total:] == -100).all()
        # The dummy trailing segment must be covered, or varlen kernels read OOB.
        assert batch["max_seqlen"] >= padding_segment
        assert batch["max_seqlen"] >= int(lengths.max())

    def test_packed_and_padded_agree_on_real_tokens(self, tokenizer):
        lengths = _true_lengths(tokenizer)
        padded = _collator(tokenizer, mlm=False)(_samples())
        packed = _collator(tokenizer, mlm=False, packing=True)(_samples())

        flat_from_padded = torch.cat(
            [padded["input_ids"][i, : int(n)] for i, n in enumerate(lengths)]
        )
        assert torch.equal(flat_from_padded, packed["input_ids"][0])


# ---------------------------------------------------------------------------
# MLM masking
# ---------------------------------------------------------------------------


class TestMLMMasking:
    @pytest.mark.parametrize("packing", [False, True])
    def test_supervised_positions_are_replaced_by_mask_token(self, tokenizer, packing):
        np.random.seed(0)
        batch = _collator(tokenizer, packing=packing, mlm_probability=0.3)(_samples())
        supervised = batch["labels"] != -100

        assert supervised.any()
        assert (batch["input_ids"][supervised] == tokenizer.mask_token_id).all()

    @pytest.mark.parametrize("packing", [False, True])
    def test_every_sequence_gets_at_least_one_masked_token(self, tokenizer, packing):
        np.random.seed(0)
        lengths = _true_lengths(tokenizer)
        # A zero rate would mask nothing without the min-one-mask guarantee.
        batch = _collator(tokenizer, packing=packing, mlm_probability=0.0)(_samples())

        if packing:
            labels = batch["labels"][0]
            offset = 0
            for n in lengths:
                n = int(n)
                assert (labels[offset : offset + n] != -100).sum() >= 1
                offset += n
        else:
            assert ((batch["labels"] != -100).sum(dim=1) >= 1).all()

    def test_special_tokens_are_never_supervised(self, tokenizer):
        np.random.seed(0)
        batch = _collator(tokenizer, mlm_probability=1.0)(_samples())
        specials = torch.tensor(sorted(tokenizer.all_special_ids))

        supervised_labels = batch["labels"][batch["labels"] != -100]
        assert not torch.isin(supervised_labels, specials).any()

    def test_labels_recover_the_unmasked_input(self, tokenizer):
        np.random.seed(0)
        clean = _collator(tokenizer, mlm=False)(_samples())["input_ids"]
        np.random.seed(0)
        masked = _collator(tokenizer, mlm_probability=0.5)(_samples())

        supervised = masked["labels"] != -100
        assert torch.equal(masked["labels"][supervised], clean[supervised])

    @pytest.mark.parametrize("masking_type", ["fixed", "beta", "cosine"])
    def test_masking_schedules_stay_in_range(self, tokenizer, masking_type):
        np.random.seed(0)
        collator = _collator(
            tokenizer, masking_type=masking_type, mlm_probability=0.15
        )
        rates = [collator._sample_mlm_probability() for _ in range(200)]

        assert all(0.0 <= r <= 1.0 for r in rates)
        if masking_type == "fixed":
            assert set(rates) == {0.15}
        else:
            assert len(set(rates)) > 1
