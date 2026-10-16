"""Regression coverage for optional model architectures and initialization."""

import importlib

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
from modules.pretrain.src.optimizer.optimizer import OptimizerConfig, get_optimizer

# The existing model tests reload this module during collection. Resolve the
# current module rather than a stale attribute on its parent package.
modeling = importlib.import_module("modules.pretrain.src.model.modeling_amplify")

NORM_TYPES = ("RMSNorm", "CenteredRMSNorm", "ZeroCenteredRMSNorm", "LayerNorm")


def tiny_config(**overrides):
    return AMPLIFYConfig(
        **{
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "max_position_embeddings": 32,
            **overrides,
        }
    )


def assert_finite_gradients(model):
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_config_round_trip(norm_type, tmp_path):
    config = modeling.AMPLIFYModelConfig(norm_type=norm_type)
    hf_config = config.to_hf_config()
    hf_config.save_pretrained(tmp_path)
    assert AMPLIFYConfig.from_pretrained(tmp_path).norm_type == norm_type
    model = modeling.AMPLIFYModel(tiny_config(norm_type=norm_type))
    expected_type = (
        nn.LayerNorm if norm_type == "LayerNorm" else getattr(modeling, norm_type)
    )
    assert type(model.norm) is expected_type
    for layer in model.layers:
        assert type(layer.input_layernorm) is expected_type
        assert type(layer.post_attention_layernorm) is expected_type


def test_invalid_norm_rejected():
    with pytest.raises(ValueError, match="norm_type"):
        AMPLIFYConfig(norm_type="invalid")
    with pytest.raises(ValidationError):
        modeling.AMPLIFYModelConfig(norm_type="invalid")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("norm_type", ["CenteredRMSNorm", "ZeroCenteredRMSNorm"])
def test_norm_numerics_and_gradients(norm_type, dtype):
    torch.manual_seed(12)
    norm = getattr(modeling, norm_type)(32, eps=1e-5)
    inputs = torch.randn(2, 8, 32, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(-0.2, 0.2, 32))
    actual = norm(inputs)
    if norm_type == "CenteredRMSNorm":
        expected = nn.functional.layer_norm(
            inputs, (32,), norm.weight.to(dtype), None, 1e-5
        )
    else:
        # An independent reference also exercises the gain's nonzero offsets.
        values = inputs.float()
        expected = (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-5)
            * (1 + norm.weight.float())
        ).to(dtype)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == dtype
    actual.float().square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_norm_initialization_and_checkpoint_round_trip(norm_type, tmp_path):
    torch.manual_seed(13)
    model = modeling.AMPLIFYForMaskedLM(tiny_config(norm_type=norm_type)).eval()
    norm = model.amplify.norm
    gain = 0.0 if norm_type == "ZeroCenteredRMSNorm" else 1.0
    torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, gain))
    ids = torch.randint(1, 32, (2, 8))
    model(ids, labels=ids).loss.backward()
    assert_finite_gradients(model)
    # Trained values must survive loading rather than being reset to defaults.
    with torch.no_grad():
        norm.weight.add_(0.125)
    expected = model(ids).logits.detach()
    model.save_pretrained(tmp_path)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
    # Missing checkpoint parameters rely on this hook, not __init__ values.
    with torch.no_grad():
        norm.weight.fill_(float("nan"))
        if getattr(norm, "bias", None) is not None:
            norm.bias.fill_(float("nan"))
    model._init_weights(norm)
    torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, gain))
    if getattr(norm, "bias", None) is not None:
        torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias))


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_checkpoint_missing_norm_parameters_are_initialized(norm_type, tmp_path):
    model = modeling.AMPLIFYForMaskedLM(tiny_config(norm_type=norm_type))
    weights = {
        name: value for name, value in model.state_dict().items() if "norm." not in name
    }
    model.save_pretrained(tmp_path, state_dict=weights)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(tmp_path)
    gain = 0.0 if norm_type == "ZeroCenteredRMSNorm" else 1.0
    for module in restored.modules():
        if isinstance(module, (modeling.RMSNorm, nn.LayerNorm)):
            torch.testing.assert_close(
                module.weight, torch.full_like(module.weight, gain)
            )
            if getattr(module, "bias", None) is not None:
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
    assert torch.isfinite(restored(torch.tensor([[3, 6, 7, 4]])).logits).all()


def test_norm_optimizer_groups_and_decay():
    model = nn.ModuleDict(
        {name: getattr(modeling, name)(32) for name in NORM_TYPES[:-1]}
    )
    model["LayerNorm"] = nn.LayerNorm(32)
    optimizer = get_optimizer(
        model,
        OptimizerConfig(
            type="AdamW",
            lr=0.1,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.2,
            fused=False,
        ),
    )
    groups = {
        group["weight_decay"]: {id(p) for p in group["params"]}
        for group in optimizer.param_groups
    }
    offset = model["ZeroCenteredRMSNorm"].weight
    assert groups[0.2] == {id(offset)}
    for name in ("RMSNorm", "CenteredRMSNorm", "LayerNorm"):
        assert id(model[name].weight) in groups[0.0]
    assert id(model["LayerNorm"].bias) in groups[0.0]
    with torch.no_grad():
        offset.fill_(0.5)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(offset, torch.full_like(offset, 0.49))
    torch.testing.assert_close(model["RMSNorm"].weight, torch.ones(32))


@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_qk_norm_precedes_rope_and_preserves_values(norm_type, packed, monkeypatch):
    torch.manual_seed(14)
    config = tiny_config(norm_type=norm_type, qk_norm=True)
    attention = modeling.Attention(config)
    norm = modeling._make_norm(config)
    expected_type = (
        nn.LayerNorm if norm_type == "LayerNorm" else getattr(modeling, norm_type)
    )
    assert type(attention.q_norm) is expected_type
    assert type(attention.k_norm) is expected_type
    assert attention.q_norm.weight.shape == (8,)
    assert attention.q_norm.weight is not attention.k_norm.weight
    with torch.no_grad():
        attention.q_norm.weight.add_(0.25)
        attention.k_norm.weight.sub_(0.125)
    hidden = torch.randn(1, 6, 32)
    cos, sin = modeling.RotaryEmbedding(config)(hidden, torch.arange(6))
    qkv = attention.qkv_proj(norm(hidden)).view(6, 3, 4, 8)
    raw_q, raw_k, raw_v = qkv.unbind(1)
    expected_q = modeling.apply_rotary_emb(
        attention.q_norm(raw_q), cos[:, None], sin[:, None]
    )
    expected_k = modeling.apply_rotary_emb(
        attention.k_norm(raw_k), cos[:, None], sin[:, None]
    )
    captured = {}

    def kernel(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v)
        return torch.zeros_like(v)

    if packed:
        attention.fa4_attn_fn = None
        monkeypatch.setattr(modeling, "varlen_attn", kernel)
        attention(
            norm,
            hidden,
            None,
            cos,
            sin,
            False,
            6,
            torch.tensor([0, 6], dtype=torch.int32),
        )
        expected_q, expected_k, raw_v = (
            t.bfloat16() for t in (expected_q, expected_k, raw_v)
        )
    else:
        monkeypatch.setattr(modeling, "scaled_dot_product_attention", kernel)
        attention(norm, hidden, None, cos[None, :, None], sin[None, :, None], False)
        expected_q, expected_k, raw_v = (
            t.transpose(0, 1)[None] for t in (expected_q, expected_k, raw_v)
        )
    torch.testing.assert_close(captured["q"], expected_q)
    torch.testing.assert_close(captured["k"], expected_k)
    torch.testing.assert_close(captured["v"], raw_v)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_qk_norm_loads_old_checkpoint_and_trains(norm_type, tmp_path):
    modeling.AMPLIFYForMaskedLM(tiny_config()).save_pretrained(tmp_path)
    model = modeling.AMPLIFYForMaskedLM.from_pretrained(
        tmp_path, qk_norm=True, norm_type=norm_type
    )
    gain = 0.0 if norm_type == "ZeroCenteredRMSNorm" else 1.0
    for layer in model.amplify.layers:
        for norm in (layer.attention.q_norm, layer.attention.k_norm):
            torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, gain))
    ids = torch.randint(1, 32, (2, 8))
    loss = model(ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert_finite_gradients(model)
    destination = tmp_path / "with_qk_norm"
    model.save_pretrained(destination)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(destination)
    torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_attention_gating_uses_normalized_inputs_before_output_projection(
    packed, monkeypatch
):
    torch.manual_seed(15)
    config = tiny_config(attention_gate=True)
    attention = modeling.Attention(config)
    norm = modeling._make_norm(config)
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    cos, sin = modeling.RotaryEmbedding(config)(hidden, torch.arange(6))
    with torch.no_grad():
        attention.o_proj.weight.copy_(2 * torch.eye(32))
        attention.gate_proj.weight.copy_(0.5 * torch.eye(32))

    def kernel(q, k, v, **kwargs):
        return torch.ones_like(v)

    if packed:
        attention.fa4_attn_fn = None
        monkeypatch.setattr(modeling, "varlen_attn", kernel)
        actual, _ = attention(
            norm,
            hidden,
            None,
            cos,
            sin,
            False,
            6,
            torch.tensor([0, 6], dtype=torch.int32),
        )
    else:
        monkeypatch.setattr(modeling, "scaled_dot_product_attention", kernel)
        actual, _ = attention(
            norm, hidden, None, cos[None, :, None], sin[None, :, None], False
        )
    expected = hidden + 2 * (0.5 * norm(hidden)).sigmoid()
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert torch.isfinite(attention.gate_proj.weight.grad).all()
    assert attention.gate_proj.weight.grad.abs().sum() > 0


def test_attention_gating_trains_and_survives_checkpoint(tmp_path):
    torch.manual_seed(16)
    config = modeling.AMPLIFYModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        attention_gate=True,
        qk_norm=True,
    )
    model = modeling.AMPLIFYForMaskedLM(config.to_hf_config())
    ids = torch.randint(1, 32, (2, 8))
    model(ids, labels=ids).loss.backward()
    assert_finite_gradients(model)
    # Exercise the explicit attention implementation as well as SDPA.
    expected = model(ids).logits
    torch.testing.assert_close(model(ids, output_attentions=True).logits, expected)
    model.save_pretrained(tmp_path)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(tmp_path)
    torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
    assert restored.config.attention_gate is True


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_nonlinear_mlm_head_trains_and_round_trips(norm_type, tmp_path):
    torch.manual_seed(17)
    config = tiny_config(
        norm_type=norm_type, nonlinear_mlm_head=True, attention_gate=True, qk_norm=True
    )
    model = modeling.AMPLIFYForMaskedLM(config)
    expected_type = (
        nn.LayerNorm if norm_type == "LayerNorm" else getattr(modeling, norm_type)
    )
    assert type(model.lm_head[2]) is expected_type
    ids = torch.randint(1, 32, (2, 8))
    labels = ids.clone()
    labels[:, 0] = -100
    output = model(ids, labels=labels)
    assert output.logits.shape == (2, 8, 32)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert_finite_gradients(model)
    model.save_pretrained(tmp_path)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(tmp_path)
    assert restored.config.nonlinear_mlm_head is True
    torch.testing.assert_close(restored(ids).logits, output.logits, rtol=0, atol=0)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_nonlinear_head_initializes_when_loading_linear_checkpoint(norm_type, tmp_path):
    modeling.AMPLIFYForMaskedLM(tiny_config(norm_type=norm_type)).save_pretrained(
        tmp_path
    )
    model = modeling.AMPLIFYForMaskedLM.from_pretrained(
        tmp_path, nonlinear_mlm_head=True
    )
    gain = 0.0 if norm_type == "ZeroCenteredRMSNorm" else 1.0
    norm = model.lm_head[2]
    torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, gain))
    if getattr(norm, "bias", None) is not None:
        torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias))
    output = model(torch.tensor([[3, 6, 7, 4]]), labels=torch.tensor([[3, 6, 7, 4]]))
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert_finite_gradients(model)


@pytest.mark.parametrize("norm_type", NORM_TYPES)
@pytest.mark.parametrize("packed", [False, True])
def test_depth_scaling_reaches_attention_gate_and_mlp(norm_type, packed, monkeypatch):
    torch.manual_seed(18)
    config = tiny_config(
        num_hidden_layers=3,
        norm_type=norm_type,
        qk_norm=True,
        attention_gate=True,
        layernorm_scaling=True,
    )
    model = modeling.AMPLIFYModel(config)
    captured = {}

    def capture(name):
        def hook(module, args):
            captured[name] = args[0].detach().clone()

        return hook

    def capture_module_input(name):
        def hook(module, args):
            captured[name] = args[1].detach().clone()

        return hook

    for i, layer in enumerate(model.layers):
        layer.attention.register_forward_pre_hook(
            capture_module_input((i, "attention_input"))
        )
        layer.mlp.register_forward_pre_hook(capture_module_input((i, "mlp_input")))
        layer.attention.qkv_proj.register_forward_pre_hook(capture((i, "qkv")))
        layer.attention.gate_proj.register_forward_pre_hook(capture((i, "gate")))
        layer.mlp.gate_up_proj.register_forward_pre_hook(capture((i, "mlp")))
        if packed:
            layer.attention.fa4_attn_fn = None

    if packed:
        monkeypatch.setattr(
            modeling, "varlen_attn", lambda q, k, v, **kwargs: torch.zeros_like(v)
        )
    ids = torch.tensor([[3, 6, 7, 8, 9, 4]])
    if packed:
        # Exercise packed blocks with a CPU kernel substitute; the public
        # model entrypoint correctly requires CUDA for packed execution.
        hidden = model.embed_tokens(ids)
        cos, sin = model.rotary_emb(hidden, torch.arange(6))
        for layer in model.layers:
            hidden, _ = layer(
                hidden,
                None,
                cos,
                sin,
                False,
                max_seqlen=6,
                cu_seqlens=torch.tensor([0, 6], dtype=torch.int32),
            )
    else:
        model(ids)
    for i, layer in enumerate(model.layers):
        scale = (i + 1) ** -0.5
        assert layer.attention.norm_scale == pytest.approx(scale)
        assert layer.mlp.norm_scale == pytest.approx(scale)
        attention_input = captured[(i, "attention_input")]
        mlp_input = captured[(i, "mlp_input")]
        if packed:
            attention_input = attention_input.flatten(0, 1)
            mlp_input = mlp_input.flatten(0, 1)
        torch.testing.assert_close(
            captured[(i, "qkv")], layer.input_layernorm(attention_input) * scale
        )
        torch.testing.assert_close(captured[(i, "gate")], captured[(i, "qkv")])
        torch.testing.assert_close(
            captured[(i, "mlp")], layer.post_attention_layernorm(mlp_input) * scale
        )


def test_depth_scaling_is_disabled_by_default():
    model = modeling.AMPLIFYModel(tiny_config())
    for layer in model.layers:
        assert layer.attention.norm_scale == 1.0
        assert layer.mlp.norm_scale == 1.0


@pytest.mark.parametrize("gaussian", [False, True])
def test_initialization_distribution_and_bias_bounds(gaussian):
    torch.manual_seed(19)
    config = tiny_config(
        gaussian_init=gaussian, decoder_init_range=0.07, embedding_init_range=0.12
    )
    model = modeling.AMPLIFYForMaskedLM(config)
    linear = nn.Linear(160, 200)
    embedding = nn.Embedding(1000, 32)
    model._init_weights(linear)
    model._init_weights(embedding)
    for weight, scale in ((linear.weight, 0.07), (embedding.weight, 0.12)):
        expected_std = scale if gaussian else scale / (3**0.5)
        assert weight.mean().item() == pytest.approx(0, abs=scale * 0.025)
        assert weight.std().item() == pytest.approx(expected_std, rel=0.04)
        if gaussian:
            assert (weight.abs() > scale).any()
        else:
            assert weight.abs().max() <= scale
    # Changing weight initialization must not change the bias policy.
    assert linear.bias.abs().max() <= 160**-0.5
    assert linear.bias.abs().max() > 0


@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_combined_variants_train_in_mixed_precision_and_round_trip(norm_type, tmp_path):
    torch.manual_seed(20)
    flags = dict(
        qk_norm=True,
        attention_gate=True,
        nonlinear_mlm_head=True,
        layernorm_scaling=True,
        gaussian_init=True,
    )
    config = modeling.AMPLIFYModelConfig(
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        vocab_size=41,
        norm_type=norm_type,
        **flags,
    )
    model = modeling.AMPLIFYForMaskedLM(config.to_hf_config())
    optimizer = get_optimizer(
        model,
        OptimizerConfig(
            type="AdamW",
            lr=0.001,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=False,
        ),
    )
    ids = torch.randint(1, 41, (2, 8))
    for _ in range(3):
        optimizer.zero_grad()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(ids, labels=ids).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert_finite_gradients(model)
        optimizer.step()
    expected = model(ids).logits
    assert expected.shape == (2, 8, 41)
    model.save_pretrained(tmp_path)
    restored = modeling.AMPLIFYForMaskedLM.from_pretrained(tmp_path)
    for flag, value in flags.items():
        assert getattr(restored.config, flag) == value
    for i, layer in enumerate(restored.amplify.layers):
        assert layer.attention.norm_scale == pytest.approx((i + 1) ** -0.5)
    torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Packed attention requires CUDA"
)
@pytest.mark.parametrize("norm_type", NORM_TYPES)
def test_combined_variants_packed_cuda_parity_and_gradients(norm_type):
    torch.manual_seed(21)
    model = (
        modeling.AMPLIFYForMaskedLM(
            tiny_config(
                hidden_size=64,
                intermediate_size=128,
                vocab_size=41,
                norm_type=norm_type,
                qk_norm=True,
                attention_gate=True,
                nonlinear_mlm_head=True,
                layernorm_scaling=True,
                gaussian_init=True,
            )
        )
        .to("cuda", dtype=torch.bfloat16)
        .eval()
    )
    ids = torch.randint(1, 41, (1, 20), device="cuda")
    positions = torch.cat((torch.arange(8), torch.arange(12))).to("cuda")
    cu_seqlens = torch.tensor([0, 8, 20], device="cuda", dtype=torch.int32)
    padded_ids = torch.zeros(2, 12, device="cuda", dtype=torch.long)
    padded_ids[0, :8] = ids[0, :8]
    padded_ids[1] = ids[0, 8:]
    mask = torch.zeros_like(padded_ids, dtype=torch.bool)
    mask[0, :8] = True
    mask[1] = True
    with torch.no_grad():
        packed = model(
            ids, position_ids=positions, cu_seqlens=cu_seqlens, max_seqlen=12
        ).logits
        padded = model(padded_ids, attention_mask=mask).logits
    torch.testing.assert_close(packed[0, :8], padded[0, :8], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(packed[0, 8:], padded[1], atol=2e-2, rtol=2e-2)
    loss = model(
        ids, labels=ids, position_ids=positions, cu_seqlens=cu_seqlens, max_seqlen=12
    ).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert_finite_gradients(model)


def test_legacy_config_keeps_optional_features_disabled():
    config = AMPLIFYConfig.from_dict(
        dict(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )
    assert config.norm_type == "RMSNorm"
    for flag in (
        "qk_norm",
        "attention_gate",
        "nonlinear_mlm_head",
        "layernorm_scaling",
        "gaussian_init",
    ):
        assert getattr(config, flag) is False
    model = modeling.AMPLIFYForMaskedLM(config)
    assert isinstance(model.lm_head, nn.Linear)
    for layer in model.amplify.layers:
        assert isinstance(layer.attention.q_norm, nn.Identity)
        assert isinstance(layer.attention.k_norm, nn.Identity)
        assert layer.attention.gate_proj is None
