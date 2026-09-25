"""Unit tests for the AMPLIFY model."""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_MODEL_DIR = _ROOT / "modules" / "pretrain" / "src" / "model"

_PKG = "modules.pretrain.src.model"
for _parent in ["modules", "modules.pretrain", "modules.pretrain.src"]:
    if _parent not in sys.modules:
        _stub = types.ModuleType(_parent)
        _stub.__path__ = [str(_ROOT / Path(*_parent.split(".")))]  # type: ignore[attr-defined]
        _stub.__package__ = _parent
        sys.modules[_parent] = _stub

if _PKG not in sys.modules:
    _pkg_stub = types.ModuleType(_PKG)
    _pkg_stub.__path__ = [str(_MODEL_DIR)]  # type: ignore[attr-defined]
    _pkg_stub.__package__ = _PKG
    sys.modules[_PKG] = _pkg_stub

_cfg_spec = importlib.util.spec_from_file_location(
    f"{_PKG}.configuration_amplify",
    _MODEL_DIR / "configuration_amplify.py",
)
_cfg_mod = importlib.util.module_from_spec(_cfg_spec)  # type: ignore[arg-type]
_cfg_mod.__package__ = _PKG
sys.modules[f"{_PKG}.configuration_amplify"] = _cfg_mod
_cfg_spec.loader.exec_module(_cfg_mod)  # type: ignore[union-attr]

_mod_spec = importlib.util.spec_from_file_location(
    f"{_PKG}.modeling_amplify",
    _MODEL_DIR / "modeling_amplify.py",
)
_mod = importlib.util.module_from_spec(_mod_spec)  # type: ignore[arg-type]
_mod.__package__ = _PKG
sys.modules[f"{_PKG}.modeling_amplify"] = _mod
_mod_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

AMPLIFYConfig = _cfg_mod.AMPLIFYConfig
RMSNorm = _mod.RMSNorm
RotaryEmbedding = _mod.RotaryEmbedding
apply_rotary_emb = _mod.apply_rotary_emb
Attention = _mod.Attention
SwiGLU = _mod.SwiGLU
EncoderBlock = _mod.EncoderBlock
AMPLIFYModel = _mod.AMPLIFYModel
AMPLIFYForMaskedLM = _mod.AMPLIFYForMaskedLM
AMPLIFYForSequenceClassification = _mod.AMPLIFYForSequenceClassification
AMPLIFYForTokenClassification = _mod.AMPLIFYForTokenClassification
get_amplify_model = _mod.get_amplify_model
get_amplify_masked_lm = _mod.get_amplify_masked_lm
get_amplify_sequence_classifier = _mod.get_amplify_sequence_classifier
get_amplify_token_classifier = _mod.get_amplify_token_classifier

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_HIDDEN = 64
_HEADS = 4
_INTERMEDIATE = 128
_VOCAB = 32
_LAYERS = 2


def _cfg(**overrides: Any) -> AMPLIFYConfig:
    defaults = dict(
        hidden_size=_HIDDEN,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        intermediate_size=_INTERMEDIATE,
        vocab_size=_VOCAB,
    )
    defaults.update(overrides)
    return AMPLIFYConfig(**defaults)


@pytest.fixture()
def cfg() -> AMPLIFYConfig:
    return _cfg()


@pytest.fixture()
def base_model(cfg: AMPLIFYConfig) -> AMPLIFYModel:
    torch.manual_seed(0)
    m = AMPLIFYModel(cfg)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# TestAMPLIFYConfig
# ---------------------------------------------------------------------------


class TestAMPLIFYConfig:
    def test_default_instantiation(self):
        c = AMPLIFYConfig()
        assert c.hidden_size == 960
        assert c.num_hidden_layers == 32
        assert c.num_attention_heads == 15
        assert c.intermediate_size == 2560
        assert c.vocab_size == 32
        assert c.max_position_embeddings == 2048
        assert c.rope_theta == 10000.0

    def test_custom_params_stored(self):
        c = _cfg(hidden_size=128, num_hidden_layers=4, rope_theta=500.0)
        assert c.hidden_size == 128
        assert c.num_hidden_layers == 4
        assert c.rope_theta == 500.0

    def test_model_type(self):
        assert AMPLIFYConfig.model_type == "amplify"

    def test_pad_bos_eos_token_ids(self):
        c = _cfg()
        assert c.pad_token_id == 0
        assert c.bos_token_id == 3
        assert c.eos_token_id == 4

    def test_norm_eps(self):
        c = _cfg()
        assert c.norm_eps == 1e-5


# ---------------------------------------------------------------------------
# TestRMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape_preserved(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).shape == x.shape

    def test_output_dtype_matches_float32(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        assert norm(x).dtype == torch.float32

    def test_weight_initialized_to_ones(self):
        norm = RMSNorm(16)
        assert torch.all(norm.weight == 1.0)

    def test_custom_eps(self):
        norm = RMSNorm(16, eps=1e-3)
        assert norm.eps == 1e-3

    def test_deterministic_output(self):
        norm = RMSNorm(32)
        x = torch.randn(2, 5, 32)
        assert torch.equal(norm(x), norm(x))


# ---------------------------------------------------------------------------
# TestRotaryEmbedding
# ---------------------------------------------------------------------------


class TestRotaryEmbedding:
    def test_table_shapes(self, cfg: AMPLIFYConfig):
        rope = RotaryEmbedding(cfg)
        head_dim = _HIDDEN // _HEADS
        assert rope.rope_cos.shape == (cfg.max_position_embeddings, head_dim // 2)
        assert rope.rope_sin.shape == (cfg.max_position_embeddings, head_dim // 2)

    def test_forward_output_shapes(self, cfg: AMPLIFYConfig):
        rope = RotaryEmbedding(cfg)
        B, S = 2, 10
        x = torch.randn(B, S, _HIDDEN)
        pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, pos)
        head_dim_half = (_HIDDEN // _HEADS) // 2
        assert cos.shape == (B, S, head_dim_half)
        assert sin.shape == (B, S, head_dim_half)

    def test_forward_deterministic(self, cfg: AMPLIFYConfig):
        rope = RotaryEmbedding(cfg)
        x = torch.randn(2, 5, _HIDDEN)
        pos = torch.arange(5).unsqueeze(0).expand(2, -1)
        cos1, sin1 = rope(x, pos)
        cos2, sin2 = rope(x, pos)
        assert torch.equal(cos1, cos2)
        assert torch.equal(sin1, sin2)


# ---------------------------------------------------------------------------
# TestApplyRotaryEmb
# ---------------------------------------------------------------------------


class TestApplyRotaryEmb:
    def test_output_shape(self):
        B, S, H, D = 2, 10, 4, 16
        x = torch.randn(B, S, H, D)
        cos = torch.randn(B, S, 1, D // 2)
        sin = torch.randn(B, S, 1, D // 2)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_deterministic(self):
        x = torch.randn(2, 5, 4, 16)
        cos = torch.randn(2, 5, 1, 8)
        sin = torch.randn(2, 5, 1, 8)
        assert torch.equal(apply_rotary_emb(x, cos, sin), apply_rotary_emb(x, cos, sin))


# ---------------------------------------------------------------------------
# TestApplyRotaryEmbQkv
# ---------------------------------------------------------------------------


class TestApplyRotaryEmbQkv:
    """`apply_rotary_emb_qkv` is the packed path's RoPE in plain torch ops."""

    def test_q_and_k_match_apply_rotary_emb_directly(self):
        total_tokens, heads, head_dim = 6, 4, 16
        qkv = torch.randn(total_tokens, 3, heads, head_dim)
        cos = torch.randn(total_tokens, head_dim // 2)
        sin = torch.randn(total_tokens, head_dim // 2)

        q, k, v = _mod.apply_rotary_emb_qkv(qkv, cos, sin)

        q_expected, k_expected, v_expected = qkv.unbind(1)
        cos_b, sin_b = cos.unsqueeze(-2), sin.unsqueeze(-2)
        assert torch.equal(q, apply_rotary_emb(q_expected, cos_b, sin_b))
        assert torch.equal(k, apply_rotary_emb(k_expected, cos_b, sin_b))
        assert torch.equal(v, v_expected)

    def test_output_shapes(self):
        total_tokens, heads, head_dim = 6, 4, 16
        qkv = torch.randn(total_tokens, 3, heads, head_dim)
        cos = torch.randn(total_tokens, head_dim // 2)
        sin = torch.randn(total_tokens, head_dim // 2)

        q, k, v = _mod.apply_rotary_emb_qkv(qkv, cos, sin)

        for t in (q, k, v):
            assert t.shape == (total_tokens, heads, head_dim)

    def test_v_is_passed_through_unrotated(self):
        total_tokens, heads, head_dim = 5, 2, 8
        qkv = torch.randn(total_tokens, 3, heads, head_dim)
        cos = torch.randn(total_tokens, head_dim // 2)
        sin = torch.randn(total_tokens, head_dim // 2)

        _, _, v = _mod.apply_rotary_emb_qkv(qkv, cos, sin)

        assert torch.equal(v, qkv[:, 2])

    def test_output_dtype_matches_qkv_not_cos_sin(self):
        # RotaryEmbedding's tables are always fp32, but q/k must follow the model dtype.
        total_tokens, heads, head_dim = 6, 4, 16
        qkv = torch.randn(total_tokens, 3, heads, head_dim, dtype=torch.bfloat16)
        cos = torch.randn(total_tokens, head_dim // 2, dtype=torch.float32)
        sin = torch.randn(total_tokens, head_dim // 2, dtype=torch.float32)

        q, k, v = _mod.apply_rotary_emb_qkv(qkv, cos, sin)

        assert q.dtype == torch.bfloat16
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16

    def test_returns_named_fields(self):
        total_tokens, heads, head_dim = 4, 2, 8
        qkv = torch.randn(total_tokens, 3, heads, head_dim)
        cos = torch.randn(total_tokens, head_dim // 2)
        sin = torch.randn(total_tokens, head_dim // 2)

        out = _mod.apply_rotary_emb_qkv(qkv, cos, sin)

        assert isinstance(out, _mod.RotaryQKV)
        assert out._fields == ("q", "k", "v")
        # q/k/v are interchangeable in shape and dtype, so the field order
        # is the only thing keeping a transposed unpack from being silent.
        q, k, v = out
        assert torch.equal(out.q, q)
        assert torch.equal(out.k, k)
        assert torch.equal(out.v, v)


# ---------------------------------------------------------------------------
# TestAttentionModule
# ---------------------------------------------------------------------------


class TestAttentionModule:
    @pytest.fixture(autouse=True)
    def _clear_fa4_loader_cache(self):
        # The function is decorated with functools.cache
        _mod._get_fa4_varlen_func.cache_clear()
        yield
        _mod._get_fa4_varlen_func.cache_clear()

    @pytest.fixture()
    def attn(self, cfg: AMPLIFYConfig) -> Attention:
        torch.manual_seed(5)
        a = Attention(cfg)
        a.eval()
        return a

    @pytest.fixture()
    def norm_and_rope(self, cfg: AMPLIFYConfig):
        norm = RMSNorm(_HIDDEN, cfg.norm_eps)
        rope = RotaryEmbedding(cfg)
        B, S = 2, 8
        x = torch.randn(B, S, _HIDDEN)
        pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, pos)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return norm, cos, sin

    def test_instantiation(self, cfg: AMPLIFYConfig):
        a = Attention(cfg)
        assert isinstance(a, nn.Module)
        assert hasattr(a, "qkv_proj")
        assert hasattr(a, "o_proj")

    def test_init_enables_fa4_on_hopper_when_installed(
        self, cfg: AMPLIFYConfig, monkeypatch: pytest.MonkeyPatch
    ):
        def fake_flash_attn_varlen(*args, **kwargs):
            return args[0]

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        monkeypatch.setattr(
            _mod, "_get_fa4_varlen_func", lambda: fake_flash_attn_varlen
        )

        a = Attention(cfg)

        assert a.fa4_attn_fn is fake_flash_attn_varlen

    def test_init_keeps_torch_varlen_below_hopper(
        self, cfg: AMPLIFYConfig, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))

        a = Attention(cfg)

        assert a.fa4_attn_fn is None

    def test_padded_forward_output_shape(self, attn: Attention, norm_and_rope):
        norm, cos, sin = norm_and_rope
        B, S = 2, 8
        h = torch.randn(B, S, _HIDDEN)
        with torch.no_grad():
            out, attn_w = attn(norm, h, None, cos, sin, output_attentions=False)
        assert out.shape == (B, S, _HIDDEN)
        assert attn_w is None

    def test_padded_forward_with_attn_weights(self, attn: Attention, norm_and_rope):
        norm, cos, sin = norm_and_rope
        B, S = 2, 8
        h = torch.randn(B, S, _HIDDEN)
        with torch.no_grad():
            out, attn_w = attn(norm, h, None, cos, sin, output_attentions=True)
        assert attn_w is not None
        assert attn_w.shape == (B, _HEADS, S, S)

    def test_residual_applied(self, attn: Attention, norm_and_rope):
        """Output should differ from the normed input — residual + projection is applied."""
        norm, cos, sin = norm_and_rope
        h = torch.randn(2, 8, _HIDDEN)
        with torch.no_grad():
            out, _ = attn(norm, h, None, cos, sin, output_attentions=False)
        assert not torch.equal(out, h)

    def test_packed_uses_torch_varlen_attention(
        self, attn: Attention, cfg: AMPLIFYConfig, monkeypatch: pytest.MonkeyPatch
    ):
        total_tokens = 6
        hidden = torch.randn(1, total_tokens, _HIDDEN)
        position_ids = torch.tensor([0, 1, 0, 1, 2, 3])
        rope = RotaryEmbedding(cfg)
        cos, sin = rope(hidden, position_ids)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        calls = []

        def fake_varlen_attn(query, key, value, **kwargs):
            calls.append((query, key, value, kwargs))
            return query

        attn.fa4_attn_fn = None
        monkeypatch.setattr(_mod, "varlen_attn", fake_varlen_attn)
        output = attn._attn_packed(
            RMSNorm(_HIDDEN, cfg.norm_eps),
            hidden,
            cos,
            sin,
            cu_seqlens,
            max_seqlen=4,
        )

        assert output.shape == hidden.shape
        assert len(calls) == 1
        query, key, value, kwargs = calls[0]
        assert (
            query.shape
            == key.shape
            == value.shape
            == (
                total_tokens,
                _HEADS,
                _HIDDEN // _HEADS,
            )
        )
        assert kwargs == {
            "cu_seq_q": cu_seqlens,
            "cu_seq_k": cu_seqlens,
            "max_q": 4,
            "max_k": 4,
        }

    def test_packed_prefers_optional_flash_attention_4(
        self, attn: Attention, cfg: AMPLIFYConfig
    ):
        total_tokens = 6
        hidden = torch.randn(1, total_tokens, _HIDDEN)
        position_ids = torch.tensor([0, 1, 0, 1, 2, 3])
        rope = RotaryEmbedding(cfg)
        cos, sin = rope(hidden, position_ids)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        calls = []

        def fake_flash_attn(query, key, value, **kwargs):
            calls.append((query, key, value, kwargs))
            return query, None

        attn.fa4_attn_fn = fake_flash_attn
        output = attn._attn_packed(
            RMSNorm(_HIDDEN, cfg.norm_eps),
            hidden,
            cos,
            sin,
            cu_seqlens,
            max_seqlen=4,
        )

        assert output.shape == hidden.shape
        assert len(calls) == 1
        _, _, _, kwargs = calls[0]
        assert kwargs == {
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": 4,
            "max_seqlen_k": 4,
            "causal": False,
        }


# ---------------------------------------------------------------------------
# TestSwiGLUModule
# ---------------------------------------------------------------------------


class TestSwiGLUModule:
    @pytest.fixture()
    def mlp(self, cfg: AMPLIFYConfig) -> SwiGLU:
        torch.manual_seed(6)
        m = SwiGLU(cfg)
        m.eval()
        return m

    def test_instantiation(self, cfg: AMPLIFYConfig):
        m = SwiGLU(cfg)
        assert isinstance(m, nn.Module)
        assert hasattr(m, "gate_up_proj")
        assert hasattr(m, "down_proj")

    def test_padded_output_shape(self, mlp: SwiGLU, cfg: AMPLIFYConfig):
        norm = RMSNorm(_HIDDEN, cfg.norm_eps)
        B, S = 2, 8
        h = torch.randn(B, S, _HIDDEN)
        with torch.no_grad():
            out = mlp(norm, h, packed=False)
        assert out.shape == (B, S, _HIDDEN)

    def test_residual_applied(self, mlp: SwiGLU, cfg: AMPLIFYConfig):
        norm = RMSNorm(_HIDDEN, cfg.norm_eps)
        h = torch.randn(2, 8, _HIDDEN)
        with torch.no_grad():
            out = mlp(norm, h, packed=False)
        assert not torch.equal(out, h)

    def test_packed_matches_padded(self, mlp: SwiGLU, cfg: AMPLIFYConfig):
        """The packed and padded paths use the same SwiGLU arithmetic."""
        norm = RMSNorm(_HIDDEN, cfg.norm_eps)
        h = torch.randn(2, 5, _HIDDEN)

        with torch.no_grad():
            packed = mlp(norm, h, packed=True)
            padded = mlp(norm, h, packed=False)

        torch.testing.assert_close(packed, padded)


# ---------------------------------------------------------------------------
# TestEncoderBlock
# ---------------------------------------------------------------------------


class TestEncoderBlock:
    @pytest.fixture()
    def block(self, cfg: AMPLIFYConfig) -> EncoderBlock:
        torch.manual_seed(1)
        b = EncoderBlock(cfg)
        b.eval()
        return b

    @pytest.fixture()
    def rope_tensors(self, cfg: AMPLIFYConfig):
        rope = RotaryEmbedding(cfg)
        B, S = 2, 10
        x = torch.randn(B, S, _HIDDEN)
        pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        cos, sin = rope(x, pos)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return cos, sin

    def test_instantiation(self, cfg: AMPLIFYConfig):
        block = EncoderBlock(cfg)
        assert isinstance(block, nn.Module)

    def test_forward_output_shape(self, block: EncoderBlock, rope_tensors):
        B, S = 2, 10
        h = torch.randn(B, S, _HIDDEN)
        cos, sin = rope_tensors
        out, _ = block(h, None, cos, sin, output_attentions=False)
        assert out.shape == (B, S, _HIDDEN)

    def test_forward_attn_weights_none_by_default(
        self, block: EncoderBlock, rope_tensors
    ):
        h = torch.randn(2, 10, _HIDDEN)
        cos, sin = rope_tensors
        _, attn_weights = block(h, None, cos, sin, output_attentions=False)
        assert attn_weights is None

    def test_forward_attn_weights_returned(self, block: EncoderBlock, rope_tensors):
        B, S = 2, 10
        h = torch.randn(B, S, _HIDDEN)
        cos, sin = rope_tensors
        _, attn_weights = block(h, None, cos, sin, output_attentions=True)
        assert attn_weights is not None
        assert attn_weights.shape == (B, _HEADS, S, S)

    def test_forward_with_attention_mask(self, block: EncoderBlock, rope_tensors):
        B, S = 2, 10
        h = torch.randn(B, S, _HIDDEN)
        cos, sin = rope_tensors
        mask = torch.ones(B, 1, 1, S, dtype=torch.bool)
        mask[0, 0, 0, -2:] = False
        out, _ = block(h, mask, cos, sin, output_attentions=False)
        assert out.shape == (B, S, _HIDDEN)

    def test_forward_deterministic(self, block: EncoderBlock, rope_tensors):
        h = torch.randn(2, 10, _HIDDEN)
        cos, sin = rope_tensors
        with torch.no_grad():
            out1, _ = block(h, None, cos, sin, output_attentions=False)
            out2, _ = block(h, None, cos, sin, output_attentions=False)
        assert torch.equal(out1, out2)

    def test_has_attention_and_mlp_submodules(self, block: EncoderBlock):
        """EncoderBlock should expose .attention and .mlp as submodules."""
        assert hasattr(block, "attention")
        assert hasattr(block, "mlp")
        assert isinstance(block.attention, Attention)
        assert isinstance(block.mlp, SwiGLU)


# ---------------------------------------------------------------------------
# TestAMPLIFYModel
# ---------------------------------------------------------------------------


class TestAMPLIFYModel:
    def test_instantiation(self, cfg: AMPLIFYConfig):
        m = AMPLIFYModel(cfg)
        assert isinstance(m, nn.Module)

    def test_forward_basic_shape(self, base_model: AMPLIFYModel):
        B, S = 2, 12
        ids = torch.randint(1, _VOCAB, (B, S))
        with torch.no_grad():
            out = base_model(ids)
        assert out.last_hidden_state.shape == (B, S, _HIDDEN)

    def test_forward_without_mask(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (3, 8))
        with torch.no_grad():
            out = base_model(ids, attention_mask=None)
        assert out.last_hidden_state.shape == (3, 8, _HIDDEN)

    def test_forward_with_boolean_mask(self, base_model: AMPLIFYModel):
        B, S = 2, 10
        ids = torch.randint(1, _VOCAB, (B, S))
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[0, -3:] = False
        with torch.no_grad():
            out = base_model(ids, attention_mask=mask)
        assert out.last_hidden_state.shape == (B, S, _HIDDEN)

    def test_forward_with_custom_position_ids(self, base_model: AMPLIFYModel):
        B, S = 2, 8
        ids = torch.randint(1, _VOCAB, (B, S))
        pos = torch.arange(S).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            out = base_model(ids, position_ids=pos)
        assert out.last_hidden_state.shape == (B, S, _HIDDEN)

    def test_output_hidden_states_count(
        self, base_model: AMPLIFYModel, cfg: AMPLIFYConfig
    ):
        ids = torch.randint(1, _VOCAB, (2, 6))
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) == cfg.num_hidden_layers + 1
        for hs in out.hidden_states:
            assert hs.shape == (2, 6, _HIDDEN)

    def test_output_hidden_states_none_by_default(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (2, 6))
        with torch.no_grad():
            out = base_model(ids, output_hidden_states=False)
        assert out.hidden_states is None

    def test_output_attentions_count(
        self, base_model: AMPLIFYModel, cfg: AMPLIFYConfig
    ):
        B, S = 2, 6
        ids = torch.randint(1, _VOCAB, (B, S))
        with torch.no_grad():
            out = base_model(ids, output_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == cfg.num_hidden_layers
        for attn in out.attentions:
            assert attn.shape == (B, _HEADS, S, S)

    def test_output_attentions_none_by_default(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (2, 6))
        with torch.no_grad():
            out = base_model(ids, output_attentions=False)
        assert out.attentions is None

    def test_forward_deterministic(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out1 = base_model(ids)
            out2 = base_model(ids)
        assert torch.equal(out1.last_hidden_state, out2.last_hidden_state)

    def test_packed_mode_without_position_ids_raises(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (1, 10))
        cu = torch.tensor([0, 5, 10], dtype=torch.int32)
        with pytest.raises(ValueError, match="position_ids"):
            base_model(ids, cu_seqlens=cu)

    def test_packed_mode_output_attentions_raises(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (1, 10))
        cu = torch.tensor([0, 5, 10], dtype=torch.int32)
        pos = torch.arange(10)
        with pytest.raises(ValueError, match="attention weights"):
            base_model(ids, cu_seqlens=cu, position_ids=pos, output_attentions=True)

    def test_packed_mode_on_cpu_raises(self, base_model: AMPLIFYModel):
        ids = torch.randint(1, _VOCAB, (1, 10))
        cu = torch.tensor([0, 5, 10], dtype=torch.int32)
        pos = torch.arange(10)
        with pytest.raises(ValueError, match="CUDA"):
            base_model(ids, cu_seqlens=cu, position_ids=pos)


# ---------------------------------------------------------------------------
# TestPackedPaddedParity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Packed mode needs a GPU.",
)
class TestPackedPaddedParity:
    """Packed varlen attention and padded SDPA must agree on real tokens."""

    def test_token_classification_with_dummy_padding_segment(self):
        torch.manual_seed(0)
        model = AMPLIFYForTokenClassification(_cfg()).cuda().eval()
        ids = torch.randint(1, _VOCAB, (1, 20), device="cuda")
        packed_ids = torch.cat(
            (ids, torch.zeros(1, 4, dtype=torch.long, device="cuda")), dim=1
        )
        cu = torch.tensor([0, 8, 20, 24], dtype=torch.int32, device="cuda")
        positions = torch.cat(
            (
                torch.arange(8, device="cuda"),
                torch.arange(12, device="cuda"),
                torch.zeros(4, device="cuda", dtype=torch.long),
            )
        ).unsqueeze(0)
        padded_ids = torch.zeros(2, 12, dtype=torch.long, device="cuda")
        padded_ids[0, :8] = ids[0, :8]
        padded_ids[1] = ids[0, 8:]
        mask = torch.zeros(2, 12, dtype=torch.bool, device="cuda")
        mask[0, :8] = True
        mask[1] = True
        with torch.no_grad():
            packed = model(
                packed_ids,
                position_ids=positions,
                cu_seqlens=cu,
                max_seqlen=12,
            ).logits
            padded = model(padded_ids, attention_mask=mask).logits
        torch.testing.assert_close(packed[0, :8], padded[0, :8], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(packed[0, 8:20], padded[1], atol=2e-2, rtol=2e-2)

    def test_fp32_sequence_classification_pools_each_packed_sequence(self):
        torch.manual_seed(0)
        model = AMPLIFYForSequenceClassification(_cfg()).cuda().eval()
        lengths = [8, 12]
        ids = torch.randint(1, _VOCAB, (1, sum(lengths)), device="cuda")
        cu = torch.tensor([0, 8, 20], dtype=torch.int32, device="cuda")
        positions = torch.cat(
            [torch.arange(n, device="cuda") for n in lengths]
        ).unsqueeze(0)
        padded_ids = torch.zeros(2, 12, dtype=torch.long, device="cuda")
        padded_ids[0, :8] = ids[0, :8]
        padded_ids[1] = ids[0, 8:]
        mask = torch.zeros(2, 12, dtype=torch.bool, device="cuda")
        mask[0, :8] = True
        mask[1] = True
        with torch.no_grad():
            packed = model(
                ids,
                position_ids=positions,
                cu_seqlens=cu,
                max_seqlen=12,
                num_sequences=2,
            ).logits
            padded = model(padded_ids, attention_mask=mask).logits
        assert packed.shape == padded.shape == (2, model.num_labels)
        torch.testing.assert_close(packed, padded, atol=2e-2, rtol=2e-2)

    def test_last_hidden_state_matches_on_real_tokens(self):
        lengths = [8, 12]
        total, longest = sum(lengths), max(lengths)
        torch.manual_seed(0)
        model = AMPLIFYModel(_cfg()).to(device="cuda", dtype=torch.bfloat16).eval()

        ids = torch.randint(1, _VOCAB, (1, total), device="cuda")
        cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device="cuda")
        pos = torch.cat([torch.arange(n) for n in lengths]).cuda()
        with torch.no_grad():
            packed = model(
                ids, cu_seqlens=cu, position_ids=pos, max_seqlen=longest
            ).last_hidden_state

        # The same tokens, re-laid-out as a padded batch. Padded position_ids
        # are derived as `arange(S)` per row, which matches each packed
        # sequence restarting at 0.
        pad_ids = torch.zeros(len(lengths), longest, dtype=ids.dtype, device="cuda")
        mask = torch.zeros(len(lengths), longest, dtype=torch.bool, device="cuda")
        start = 0
        for i, n in enumerate(lengths):
            pad_ids[i, :n] = ids[0, start : start + n]
            mask[i, :n] = True
            start += n
        with torch.no_grad():
            padded = model(pad_ids, attention_mask=mask).last_hidden_state

        start = 0
        for i, n in enumerate(lengths):
            torch.testing.assert_close(
                packed[0, start : start + n].float(),
                padded[i, :n].float(),
                atol=2e-2,
                rtol=2e-2,
            )
            start += n


# ---------------------------------------------------------------------------
# TestAMPLIFYForMaskedLM
# ---------------------------------------------------------------------------


class TestAMPLIFYForMaskedLM:
    @pytest.fixture()
    def model(self, cfg: AMPLIFYConfig) -> AMPLIFYForMaskedLM:
        torch.manual_seed(2)
        m = AMPLIFYForMaskedLM(cfg)
        m.eval()
        return m

    def test_logits_shape(self, model: AMPLIFYForMaskedLM):
        B, S = 2, 10
        ids = torch.randint(1, _VOCAB, (B, S))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (B, S, _VOCAB)

    def test_loss_none_without_labels(self, model: AMPLIFYForMaskedLM):
        ids = torch.randint(1, _VOCAB, (2, 10))
        with torch.no_grad():
            out = model(ids)
        assert out.loss is None

    def test_loss_scalar_with_labels(self, model: AMPLIFYForMaskedLM):
        B, S = 2, 10
        ids = torch.randint(1, _VOCAB, (B, S))
        labels = ids.clone()
        labels[:, :5] = -100
        with torch.no_grad():
            out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()
        assert out.loss.item() > 0

    def test_hidden_states_none_by_default(self, model: AMPLIFYForMaskedLM):
        ids = torch.randint(1, _VOCAB, (2, 6))
        with torch.no_grad():
            out = model(ids)
        assert out.hidden_states is None

    def test_hidden_states_returned(
        self, model: AMPLIFYForMaskedLM, cfg: AMPLIFYConfig
    ):
        ids = torch.randint(1, _VOCAB, (2, 6))
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) == cfg.num_hidden_layers + 1


# ---------------------------------------------------------------------------
# TestAMPLIFYForSequenceClassification
# ---------------------------------------------------------------------------


class TestAMPLIFYForSequenceClassification:
    @pytest.fixture()
    def model_clf(self) -> AMPLIFYForSequenceClassification:
        torch.manual_seed(3)
        m = AMPLIFYForSequenceClassification(_cfg(num_labels=3))
        m.eval()
        return m

    @pytest.fixture()
    def model_reg(self) -> AMPLIFYForSequenceClassification:
        torch.manual_seed(3)
        m = AMPLIFYForSequenceClassification(_cfg(num_labels=1))
        m.eval()
        return m

    def test_logits_shape_classification(
        self, model_clf: AMPLIFYForSequenceClassification
    ):
        ids = torch.randint(1, _VOCAB, (4, 10))
        with torch.no_grad():
            out = model_clf(ids)
        assert out.logits.shape == (4, 3)

    def test_logits_shape_regression(self, model_reg: AMPLIFYForSequenceClassification):
        ids = torch.randint(1, _VOCAB, (4, 10))
        with torch.no_grad():
            out = model_reg(ids)
        assert out.logits.shape == (4, 1)

    def test_loss_cross_entropy_with_labels(
        self, model_clf: AMPLIFYForSequenceClassification
    ):
        ids = torch.randint(1, _VOCAB, (4, 10))
        labels = torch.randint(0, 3, (4,))
        with torch.no_grad():
            out = model_clf(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()

    def test_loss_mse_regression(self, model_reg: AMPLIFYForSequenceClassification):
        ids = torch.randint(1, _VOCAB, (4, 10))
        labels = torch.randn(4)
        with torch.no_grad():
            out = model_reg(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()

    def test_loss_none_without_labels(
        self, model_clf: AMPLIFYForSequenceClassification
    ):
        ids = torch.randint(1, _VOCAB, (4, 10))
        with torch.no_grad():
            out = model_clf(ids)
        assert out.loss is None

    def test_mean_pooling_with_mask(self, model_clf: AMPLIFYForSequenceClassification):
        B, S = 2, 10
        ids = torch.randint(1, _VOCAB, (B, S))
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[0, -3:] = False
        with torch.no_grad():
            out = model_clf(ids, attention_mask=mask)
        assert out.logits.shape == (B, 3)

    def test_mean_pooling_without_mask(
        self, model_clf: AMPLIFYForSequenceClassification
    ):
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out = model_clf(ids, attention_mask=None)
        assert out.logits.shape == (2, 3)


# ---------------------------------------------------------------------------
# TestAMPLIFYForTokenClassification
# ---------------------------------------------------------------------------


class TestAMPLIFYForTokenClassification:
    @pytest.fixture()
    def model(self) -> AMPLIFYForTokenClassification:
        torch.manual_seed(4)
        m = AMPLIFYForTokenClassification(_cfg(num_labels=5))
        m.eval()
        return m

    def test_logits_shape(self, model: AMPLIFYForTokenClassification):
        B, S = 2, 12
        ids = torch.randint(1, _VOCAB, (B, S))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (B, S, 5)

    def test_loss_none_without_labels(self, model: AMPLIFYForTokenClassification):
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out = model(ids)
        assert out.loss is None

    def test_loss_scalar_with_labels(self, model: AMPLIFYForTokenClassification):
        B, S = 2, 8
        ids = torch.randint(1, _VOCAB, (B, S))
        labels = torch.randint(0, 5, (B, S))
        with torch.no_grad():
            out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()

    def test_loss_with_ignored_positions(self, model: AMPLIFYForTokenClassification):
        B, S = 2, 8
        ids = torch.randint(1, _VOCAB, (B, S))
        labels = torch.randint(0, 5, (B, S))
        labels[:, -2:] = -100
        with torch.no_grad():
            out = model(ids, labels=labels)
        assert out.loss is not None
        assert out.loss.shape == ()


# ---------------------------------------------------------------------------
# TestNumericalEquivalence
# ---------------------------------------------------------------------------


class TestNumericalEquivalence:
    def test_same_weights_same_output(self):
        torch.manual_seed(42)
        model = AMPLIFYModel(_cfg())
        model.eval()
        ids = torch.randint(1, _VOCAB, (2, 10))
        with torch.no_grad():
            out1 = model(ids)
            out2 = model(ids)
        assert torch.equal(out1.last_hidden_state, out2.last_hidden_state)

    def test_state_dict_reload_preserves_output(self):
        torch.manual_seed(42)
        model = AMPLIFYModel(_cfg())
        model.eval()
        ids = torch.randint(1, _VOCAB, (2, 10))
        with torch.no_grad():
            out_before = model(ids).last_hidden_state

        sd = model.state_dict()
        torch.manual_seed(99)
        model2 = AMPLIFYModel(_cfg())
        model2.load_state_dict(sd)
        model2.eval()
        with torch.no_grad():
            out_after = model2(ids).last_hidden_state

        assert torch.equal(out_before, out_after)


# ---------------------------------------------------------------------------
# TestAMPLIFYModelConfig
# ---------------------------------------------------------------------------

AMPLIFYModelConfig = _mod.AMPLIFYModelConfig


class TestAMPLIFYModelConfig:
    def test_default_instantiation(self):
        cfg = AMPLIFYModelConfig()
        assert cfg.hidden_size == 960
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 15
        assert cfg.intermediate_size == 2560
        assert cfg.vocab_size == 32
        assert cfg.max_position_embeddings == 2048
        assert cfg.rope_theta == 10000.0
        assert cfg.num_labels == 2

    def test_custom_params(self):
        cfg = AMPLIFYModelConfig(hidden_size=128, num_hidden_layers=4, num_labels=5)
        assert cfg.hidden_size == 128
        assert cfg.num_hidden_layers == 4
        assert cfg.num_labels == 5

    def test_extra_fields_forbidden(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            AMPLIFYModelConfig(unknown_field=42)

    def test_to_hf_config_returns_amplify_config(self):
        cfg = AMPLIFYModelConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=32,
        )
        hf = cfg.to_hf_config()
        assert isinstance(hf, AMPLIFYConfig)
        assert hf.hidden_size == 64
        assert hf.num_hidden_layers == 2

    def test_num_labels_forwarded_to_hf_config(self):
        cfg = AMPLIFYModelConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            vocab_size=32,
            num_labels=7,
        )
        hf = cfg.to_hf_config()
        assert hf.num_labels == 7

    def test_get_amplify_model_returns_amplify_model(self):
        cfg = AMPLIFYModelConfig(**dict(hidden_size=_HIDDEN, num_hidden_layers=_LAYERS,
                                       num_attention_heads=_HEADS,
                                       intermediate_size=_INTERMEDIATE, vocab_size=_VOCAB))
        model = get_amplify_model(cfg)
        assert isinstance(model, AMPLIFYModel)

    def test_get_amplify_masked_lm(self):
        cfg = AMPLIFYModelConfig(**dict(hidden_size=_HIDDEN, num_hidden_layers=_LAYERS,
                                       num_attention_heads=_HEADS,
                                       intermediate_size=_INTERMEDIATE, vocab_size=_VOCAB))
        model = get_amplify_masked_lm(cfg)
        assert isinstance(model, AMPLIFYForMaskedLM)
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (2, 8, _VOCAB)

    def test_get_amplify_sequence_classifier(self):
        cfg = AMPLIFYModelConfig(**dict(hidden_size=_HIDDEN, num_hidden_layers=_LAYERS,
                                       num_attention_heads=_HEADS,
                                       intermediate_size=_INTERMEDIATE,
                                       vocab_size=_VOCAB, num_labels=3))
        model = get_amplify_sequence_classifier(cfg)
        assert isinstance(model, AMPLIFYForSequenceClassification)
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (2, 3)

    def test_get_amplify_token_classifier(self):
        cfg = AMPLIFYModelConfig(**dict(hidden_size=_HIDDEN, num_hidden_layers=_LAYERS,
                                       num_attention_heads=_HEADS,
                                       intermediate_size=_INTERMEDIATE,
                                       vocab_size=_VOCAB, num_labels=4))
        model = get_amplify_token_classifier(cfg)
        assert isinstance(model, AMPLIFYForTokenClassification)
        ids = torch.randint(1, _VOCAB, (2, 8))
        with torch.no_grad():
            out = model(ids)
        assert out.logits.shape == (2, 8, 4)


# ---------------------------------------------------------------------------
# TestCheckpointRoundTrip
#
# Regression tests for the RoPE table bug: the cos/sin tables are non-persistent
# buffers, so they never land in a checkpoint, and `from_pretrained` reallocates
# them with `torch.empty_like` (uninitialised memory). Before the fix a reloaded
# checkpoint silently produced NaN logits.
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_rope_tables_absent_from_state_dict(self, cfg: AMPLIFYConfig):
        model = AMPLIFYForMaskedLM(cfg)
        assert not [k for k in model.state_dict() if "rope" in k]

    def test_init_weights_rebuilds_clobbered_rope_tables(self, cfg: AMPLIFYConfig):
        model = AMPLIFYForMaskedLM(cfg)
        rope = model.amplify.rotary_emb
        expected_cos = rope.rope_cos.clone()
        expected_sin = rope.rope_sin.clone()

        rope.rope_cos = torch.full_like(rope.rope_cos, float("nan"))
        rope.rope_sin = torch.full_like(rope.rope_sin, float("nan"))
        model._init_weights(rope)

        assert torch.equal(rope.rope_cos, expected_cos)
        assert torch.equal(rope.rope_sin, expected_sin)

    def test_init_weights_initializes_clobbered_linear_bias(self, cfg: AMPLIFYConfig):
        """Regression: `_init_weights` must fill biases, not just weights.

        `from_pretrained` allocates parameters missing from a checkpoint with
        `torch.empty` and relies solely on this hook to fill them, so a bias
        left untouched keeps uninitialised memory.
        """
        model = AMPLIFYForMaskedLM(cfg)
        head = model.lm_head
        head.bias.data = torch.full_like(head.bias, float("nan"))

        model._init_weights(head)

        bound = 1.0 / math.sqrt(head.weight.shape[1])
        assert torch.isfinite(head.bias).all()
        assert head.bias.abs().max() <= bound

    def test_init_weights_leaves_bias_free_trunk_untouched(self, cfg: AMPLIFYConfig):
        model = AMPLIFYForMaskedLM(cfg)
        biased = [
            name
            for name, mod in model.amplify.named_modules()
            if isinstance(mod, nn.Linear) and mod.bias is not None
        ]
        assert biased == []

    @pytest.mark.parametrize(
        "head_cls, num_labels",
        [
            (AMPLIFYForSequenceClassification, 2),
            (AMPLIFYForTokenClassification, 3),
        ],
    )
    def test_fresh_task_head_bias_is_initialized(
        self, cfg: AMPLIFYConfig, tmp_path, head_cls, num_labels
    ):
        """A head absent from the checkpoint must not inherit garbage memory."""
        AMPLIFYForMaskedLM(cfg).save_pretrained(tmp_path)

        model = head_cls.from_pretrained(
            tmp_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        bias = model.classifier.bias
        bound = 1.0 / math.sqrt(model.classifier.weight.shape[1])
        assert bias.shape == (num_labels,)
        assert torch.isfinite(bias).all()
        # Two-sided on purpose: an uninitialised bias reads back as exactly zero
        # on CPU (fresh pages are zero-filled) but as ~1e9 garbage on CUDA, so
        # only bounding both ends detects the regression on either platform.
        assert bias.abs().max() <= bound
        assert bias.abs().max() > 0.0

    def test_fresh_task_head_loss_is_near_uniform_prior(
        self, cfg: AMPLIFYConfig, tmp_path
    ):
        """An untrained head starts near ln(num_labels), not a saturated logit.

        A corrupted bias silently produced losses in the 1e10 range, which zeroed
        gradients and pinned predictions to a single class.
        """
        num_labels = 4
        AMPLIFYForMaskedLM(cfg).save_pretrained(tmp_path)
        model = AMPLIFYForSequenceClassification.from_pretrained(
            tmp_path, num_labels=num_labels, ignore_mismatched_sizes=True
        ).eval()

        ids = torch.randint(1, _VOCAB, (4, 16))
        mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.no_grad():
            out = model(
                input_ids=ids,
                attention_mask=mask,
                labels=torch.arange(num_labels),
            )

        assert torch.isfinite(out.loss)
        assert out.loss.item() == pytest.approx(math.log(num_labels), abs=0.5)

    def test_from_pretrained_matches_live_model(self, cfg: AMPLIFYConfig, tmp_path):
        torch.manual_seed(0)
        model = AMPLIFYForMaskedLM(cfg).eval()
        ids = torch.randint(1, _VOCAB, (2, 16))
        with torch.no_grad():
            expected = model(input_ids=ids, labels=ids.clone()).loss
        model.save_pretrained(tmp_path)
        del model

        # Poison the allocator so the memory handed back to `torch.empty_like`
        # during loading is NaN rather than a stale copy of the correct table.
        junk = [
            torch.full((cfg.max_position_embeddings, 32), float("nan"))
            for _ in range(200)
        ]
        del junk

        reloaded = AMPLIFYForMaskedLM.from_pretrained(tmp_path).eval()
        rope = reloaded.amplify.rotary_emb
        assert rope.rope_cos[0, 0].item() == pytest.approx(1.0)
        assert rope.rope_sin[0, 0].item() == pytest.approx(0.0)

        with torch.no_grad():
            actual = reloaded(input_ids=ids, labels=ids.clone()).loss
        assert torch.isfinite(actual)
        assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# TestAutoClassRegistration
#
# `save_pretrained` only copies the defining source files into a checkpoint (and
# only records an `auto_map`) for classes registered for an auto class. Without
# it, checkpoints are not self-contained and `AutoModelForMaskedLM.from_pretrained`
# fails for any consumer that doesn't have this repo importable.
# ---------------------------------------------------------------------------


register_amplify_auto_classes = _mod.register_amplify_auto_classes


class TestAutoClassRegistration:
    def test_auto_map_covers_every_head(self):
        auto_map = register_amplify_auto_classes()
        assert auto_map == {
            "AutoConfig": "configuration_amplify.AMPLIFYConfig",
            "AutoModel": "modeling_amplify.AMPLIFYModel",
            "AutoModelForMaskedLM": "modeling_amplify.AMPLIFYForMaskedLM",
            "AutoModelForSequenceClassification": "modeling_amplify.AMPLIFYForSequenceClassification",
            "AutoModelForTokenClassification": "modeling_amplify.AMPLIFYForTokenClassification",
        }

    def test_classes_are_registered_for_their_auto_class(self):
        register_amplify_auto_classes()
        assert AMPLIFYConfig._auto_class == "AutoConfig"
        assert AMPLIFYModel._auto_class == "AutoModel"
        assert AMPLIFYForMaskedLM._auto_class == "AutoModelForMaskedLM"
        assert AMPLIFYForSequenceClassification._auto_class == (
            "AutoModelForSequenceClassification"
        )
        assert AMPLIFYForTokenClassification._auto_class == (
            "AutoModelForTokenClassification"
        )

    def test_save_pretrained_bundles_source_files(self, cfg: AMPLIFYConfig, tmp_path):
        model = AMPLIFYForMaskedLM(cfg)
        model.config.auto_map = register_amplify_auto_classes()
        model.save_pretrained(tmp_path)

        written = {p.name for p in tmp_path.iterdir()}
        # Both files are required: modeling_amplify.py relative-imports the config.
        assert {"modeling_amplify.py", "configuration_amplify.py"} <= written

        import json

        saved_map = json.loads((tmp_path / "config.json").read_text())["auto_map"]
        assert saved_map["AutoModelForMaskedLM"] == (
            "modeling_amplify.AMPLIFYForMaskedLM"
        )
        assert saved_map["AutoModel"] == "modeling_amplify.AMPLIFYModel"
