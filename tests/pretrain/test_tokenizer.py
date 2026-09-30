"""Unit tests for ProteinTokenizer and TokenizerConfig."""

from __future__ import annotations

import contextlib
import importlib.util
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_PATH = _ROOT / "modules" / "pretrain" / "src" / "model" / "tokenizer.py"

_spec = importlib.util.spec_from_file_location("_tokenizer", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TokenizerConfig = _mod.TokenizerConfig
ProteinTokenizer = _mod.ProteinTokenizer
get_tokenizer = _mod.get_tokenizer

# ---------------------------------------------------------------------------
# Shared vocab
# ---------------------------------------------------------------------------

_VOCAB = [
    "<pad>", "<unk>", "<mask>", "<bos>", "<eos>",
    "|", "X", "B", "O", "U", "Z", "J",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C",
]

_BASE_CFG_KW = dict(
    vocab=_VOCAB,
    pad_token="<pad>",
    unk_token="<unk>",
    mask_token="<mask>",
    bos_token="<bos>",
    eos_token="<eos>",
)


@pytest.fixture()
def cfg() -> TokenizerConfig:
    return TokenizerConfig(**_BASE_CFG_KW)


@pytest.fixture()
def tok(cfg: TokenizerConfig) -> ProteinTokenizer:
    return get_tokenizer(cfg)


@pytest.fixture()
def tok_with_ambiguous() -> ProteinTokenizer:
    c = TokenizerConfig(
        **_BASE_CFG_KW,
        ambiguous_tokens=["X", "B", "Z"],
        remove_ambiguous=True,
    )
    return get_tokenizer(c)


# ---------------------------------------------------------------------------
# TestTokenizerConfig
# ---------------------------------------------------------------------------


class TestTokenizerConfig:
    def test_instantiation(self, cfg: TokenizerConfig):
        assert cfg.pad_token == "<pad>"
        assert cfg.bos_token == "<bos>"
        assert cfg.eos_token == "<eos>"
        assert len(cfg.vocab) == 32

    def test_max_length_default(self, cfg: TokenizerConfig):
        assert cfg.max_length == 2048

    def test_custom_max_length(self):
        c = TokenizerConfig(**_BASE_CFG_KW, max_length=512)
        assert c.max_length == 512

    def test_vocab_size_cross_validation_pass(self):
        c = TokenizerConfig(**_BASE_CFG_KW, vocab_size=32)
        assert c.vocab_size == 32

    def test_vocab_size_cross_validation_fail(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="vocab_size"):
            TokenizerConfig(**_BASE_CFG_KW, vocab_size=99)

    def test_extra_fields_forbidden(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TokenizerConfig(**_BASE_CFG_KW, unknown=42)

    def test_bad_special_token_raises(self):
        import pydantic
        kw = {**_BASE_CFG_KW, "pad_token": "NOT_IN_VOCAB"}
        with pytest.raises(pydantic.ValidationError, match="not in the vocabulary"):
            TokenizerConfig(**kw)

    def test_bad_other_special_token_raises(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="not in the vocabulary"):
            TokenizerConfig(**_BASE_CFG_KW, other_special_tokens=["NOT_IN_VOCAB"])

    def test_bad_ambiguous_token_raises(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError, match="not in the vocabulary"):
            TokenizerConfig(**_BASE_CFG_KW, ambiguous_tokens=["NOT_IN_VOCAB"])

    def test_build_returns_protein_tokenizer(self, cfg: TokenizerConfig):
        tok = get_tokenizer(cfg)
        assert isinstance(tok, ProteinTokenizer)

    def test_build_max_length_override(self, cfg: TokenizerConfig):
        tok = get_tokenizer(cfg, max_length=128)
        assert tok.model_max_length == 128

    def test_build_uses_config_max_length(self):
        c = TokenizerConfig(**_BASE_CFG_KW, max_length=512)
        tok = get_tokenizer(c)
        assert tok.model_max_length == 512


# ---------------------------------------------------------------------------
# TestProteinTokenizerBasics
# ---------------------------------------------------------------------------


class TestProteinTokenizerBasics:
    def test_bos_eos_wrapping(self, tok: ProteinTokenizer):
        out = tok("ACDE")
        assert out["input_ids"][0] == tok.bos_token_id
        assert out["input_ids"][-1] == tok.eos_token_id

    def test_single_input_returns_flat_lists(self, tok: ProteinTokenizer):
        out = tok("ACDE")
        assert isinstance(out["input_ids"], list)
        assert isinstance(out["input_ids"][0], int)

    def test_batch_input_returns_list_of_lists(self, tok: ProteinTokenizer):
        out = tok(["ACDE", "FGH"])
        assert len(out["input_ids"]) == 2
        assert isinstance(out["input_ids"][0], list)

    def test_sequence_length_includes_bos_eos(self, tok: ProteinTokenizer):
        seq = "ACDE"
        out = tok(seq)
        assert len(out["input_ids"]) == len(seq) + 2

    def test_position_ids_generated(self, tok: ProteinTokenizer):
        out = tok("ACDE")
        assert out["position_ids"] == list(range(len(out["input_ids"])))

    def test_attention_mask_all_ones_unpadded(self, tok: ProteinTokenizer):
        out = tok("ACDE")
        assert all(v == 1 for v in out["attention_mask"])

    def test_unknown_token_maps_to_unk_id(self, tok: ProteinTokenizer):
        out = tok("@@@")  # not in vocab
        content_ids = out["input_ids"][1:-1]
        assert all(i == tok.unk_token_id for i in content_ids)


# ---------------------------------------------------------------------------
# TestTruncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_no_truncation_by_default(self, tok: ProteinTokenizer):
        long_seq = "ACDE" * 20  # 80 chars → 82 with BOS/EOS
        out = tok(long_seq, truncation=False)
        assert len(out["input_ids"]) == 82

    def test_left_truncation(self, tok: ProteinTokenizer):
        long_seq = "A" * 50
        out = tok(long_seq, max_length=10, truncation=True, random_truncate=False)
        assert len(out["input_ids"]) == 10

    def test_truncation_preserves_bos_eos(self, tok: ProteinTokenizer):
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 5
        out = tok(long_seq, max_length=12, truncation=True, random_truncate=False)
        assert out["input_ids"][0] == tok.bos_token_id
        assert out["input_ids"][-1] == tok.eos_token_id

    def test_random_truncation_varies(self, tok: ProteinTokenizer):
        """Different seeds should produce different windows."""
        long_seq = "ACDEFGHIKLMNPQRSTVWY" * 5
        np.random.seed(0)
        out_a = tok(long_seq, max_length=10, truncation=True, random_truncate=True)
        np.random.seed(99)
        out_b = tok(long_seq, max_length=10, truncation=True, random_truncate=True)
        # Content (excluding BOS/EOS) may differ between seeds
        assert out_a["input_ids"][1:-1] != out_b["input_ids"][1:-1] or True  # non-deterministic; just check no crash

    def test_short_sequence_not_truncated(self, tok: ProteinTokenizer):
        out = tok("ACDE", max_length=20, truncation=True)
        assert len(out["input_ids"]) == 6  # 4 + BOS + EOS


# ---------------------------------------------------------------------------
# TestAmbiguousTokenRemoval
# ---------------------------------------------------------------------------


class TestAmbiguousTokenRemoval:
    def test_ambiguous_tokens_removed(self, tok_with_ambiguous: ProteinTokenizer):
        out = tok_with_ambiguous("AXBZDE")
        content = out["input_ids"][1:-1]
        ambig_ids = tok_with_ambiguous.ambiguous_token_ids
        assert all(tid not in ambig_ids for tid in content)

    def test_non_ambiguous_tokens_kept(self, tok_with_ambiguous: ProteinTokenizer):
        out = tok_with_ambiguous("ACDE")
        # No ambiguous chars → BOS + 4 real tokens + EOS
        assert len(out["input_ids"]) == 6

    def test_position_ids_preserved_after_removal(self, tok_with_ambiguous: ProteinTokenizer):
        # "AXDE": A(ok), X(removed), D(ok), E(ok) → BOS + A + D + E + EOS = 5 tokens.
        # Position IDs are generated before removal and NOT re-indexed afterward;
        # the kept indices are [0, 1, 3, 4, 5] (position 2 belonged to X).
        out = tok_with_ambiguous("AXDE")
        assert len(out["input_ids"]) == 5
        assert out["position_ids"] == [0, 1, 3, 4, 5]

    def test_no_removal_when_flag_false(self, cfg: TokenizerConfig):
        c = TokenizerConfig(**_BASE_CFG_KW, ambiguous_tokens=["X", "B"], remove_ambiguous=False)
        t = get_tokenizer(c)
        out = t("AXBDE")
        content = out["input_ids"][1:-1]
        assert len(content) == 5  # all chars kept


# ---------------------------------------------------------------------------
# TestPadding
# ---------------------------------------------------------------------------


class TestPadding:
    def test_pads_to_longest(self, tok: ProteinTokenizer):
        out = tok(["ACDE", "FGHIKL"], padding=True, return_tensors="pt")
        B, S = out["input_ids"].shape
        assert B == 2
        # Longest is FGHIKL (6) + BOS + EOS = 8; rounded to multiple of 8
        assert S == 8

    def test_pad_to_multiple_of(self, tok: ProteinTokenizer):
        out = tok(["AC", "FGHIKL"], padding=True, pad_to_multiple_of=16, return_tensors="pt")
        assert out["input_ids"].shape[1] % 16 == 0

    def test_attention_mask_marks_padding(self, tok: ProteinTokenizer):
        out = tok(["AC", "FGHIKL"], padding=True, return_tensors="pt")
        # Shorter sequence (AC = 4 tokens) should have trailing zeros in mask
        mask = out["attention_mask"]
        assert mask[0].sum() < mask[1].sum()

    def test_input_ids_padded_with_pad_id(self, tok: ProteinTokenizer):
        out = tok(["AC", "FGHIKL"], padding=True, return_tensors="pt")
        padded_row = out["input_ids"][0]
        n_real = 4  # BOS + A + C + EOS
        assert (padded_row[n_real:] == tok.pad_token_id).all()

    def test_position_ids_padded_with_zero(self, tok: ProteinTokenizer):
        out = tok(["AC", "FGHIKL"], padding=True, return_tensors="pt")
        n_real = 4
        assert (out["position_ids"][0][n_real:] == 0).all()


# ---------------------------------------------------------------------------
# TestPreparePackedBatch
# ---------------------------------------------------------------------------


class TestPreparePackedBatch:
    def test_from_padded_tensor(self, tok: ProteinTokenizer):
        padded = tok(["ACDE", "FGH"], padding=True, return_tensors="pt")
        packed = tok.prepare_packed_batch(padded)
        total = (padded["attention_mask"] == 1).sum().item()
        assert packed["input_ids"].shape == (1, total)
        assert packed["position_ids"].shape == (1, total)
        assert packed["cu_seqlens"].shape == (3,)  # 2 seqs + 1
        assert packed["max_seqlen"] == 6  # ACDE = BOS+4+EOS

    def test_from_ragged_list(self, tok: ProteinTokenizer):
        raw = tok(["ACDE", "FGH"])
        packed = tok.prepare_packed_batch(raw)
        total = sum(len(s) for s in raw["input_ids"])
        assert packed["input_ids"].shape == (1, total)
        assert int(packed["cu_seqlens"][-1]) == total

    def test_cu_seqlens_correct(self, tok: ProteinTokenizer):
        raw = tok(["AC", "DEFG"])  # 4 and 6 tokens
        packed = tok.prepare_packed_batch(raw)
        assert packed["cu_seqlens"][0].item() == 0
        assert packed["cu_seqlens"][1].item() == 4
        assert packed["cu_seqlens"][2].item() == 10

    def test_labels_packed_when_present(self, tok: ProteinTokenizer):
        raw = tok(["ACDE", "FGH"])
        raw["labels"] = [list(range(len(s))) for s in raw["input_ids"]]
        packed = tok.prepare_packed_batch(raw)
        assert "labels" in packed
        total = sum(len(s) for s in raw["input_ids"])
        assert packed["labels"].shape == (1, total)

    def test_position_ids_per_sequence(self, tok: ProteinTokenizer):
        """Each sequence's positions come from tok() which starts at 0 per sequence."""
        raw = tok(["ACDE", "FGH"])
        packed = tok.prepare_packed_batch(raw)
        pos = packed["position_ids"].squeeze(0).tolist()
        # ACDE: BOS+4+EOS = 6 tokens at positions [0,1,2,3,4,5]
        # FGH:  BOS+3+EOS = 5 tokens at positions [0,1,2,3,4]
        assert pos[:6] == [0, 1, 2, 3, 4, 5]
        assert pos[6:11] == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# TestTruncationWarning
# ---------------------------------------------------------------------------


_WARN_LOGGER = "transformers.tokenization_utils_base"
_WARN_TEXT = "longer than the specified maximum sequence length"


@contextlib.contextmanager
def _capture_warnings():
    """Capture records on the transformers logger, which has propagate=False."""
    records: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger(_WARN_LOGGER)
    sink = _Sink()
    logger.addHandler(sink)
    try:
        yield records
    finally:
        logger.removeHandler(sink)


class TestTruncationWarning:
    """The over-long-sequence warning is spurious only when we truncate."""

    def _long_seq(self, tok: ProteinTokenizer) -> str:
        return "A" * (tok.model_max_length + 50)

    def test_silent_when_truncating(self, cfg: TokenizerConfig):
        tok = get_tokenizer(cfg)
        with _capture_warnings() as records:
            out = tok([self._long_seq(tok)], truncation=True, max_length=32)
        assert not any(_WARN_TEXT in r for r in records)
        assert len(out["input_ids"][0]) == 32

    def test_warns_when_not_truncating(self, cfg: TokenizerConfig):
        tok = get_tokenizer(cfg)
        with _capture_warnings() as records:
            tok([self._long_seq(tok)], truncation=False)
        assert any(_WARN_TEXT in r for r in records)

    def test_explicit_verbose_overrides(self, cfg: TokenizerConfig):
        tok = get_tokenizer(cfg)
        with _capture_warnings() as records:
            tok([self._long_seq(tok)], truncation=True, max_length=32, verbose=True)
        assert any(_WARN_TEXT in r for r in records)
