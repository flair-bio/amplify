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
