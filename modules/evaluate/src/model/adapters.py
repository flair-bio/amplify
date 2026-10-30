"""Backbone compatibility for original-coordinate contact inference."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def masked_lm_logits(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Autocast a supported encoder, then run its public MLM head in FP32."""
    device = inputs["input_ids"].device
    encoder = getattr(model, "base_model", None)
    head = getattr(model, "lm_head", None)
    if encoder is not None and encoder is not model and head is not None:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = encoder(**inputs)
            hidden = output.last_hidden_state
        with torch.autocast(device.type, enabled=False):
            return head(hidden.float())
    # Unknown wrappers retain full-precision semantics instead of being patched.
    output = model(**inputs)
    return output.logits if hasattr(output, "logits") else output[0]


@contextmanager
def contact_positions(model: Any, positions: torch.Tensor) -> Iterator[None]:
    """Prefer position_ids; adapt rotary tables when needed and restore state."""
    rotary = [
        module
        for module in model.modules()
        if all(
            hasattr(module, name)
            for name in ("_update_cos_sin_tables", "_seq_len_cached")
        )
    ]
    saved = []
    grown = []
    legacy_table = getattr(model, "freqs_cis", None)
    try:
        # Table-based AMPLIFY exports expose a device-specific table builder.
        for module in model.modules():
            reset = getattr(module, "reset_tables", None)
            build = getattr(module, "_build_tables", None)
            if all(
                hasattr(module, name)
                for name in ("rope_cos", "rope_sin", "max_position_embeddings")
            ) and (callable(reset) or callable(build)):
                length = int(positions.max()) + 1
                if length > module.max_position_embeddings:
                    old_cos, old_sin = module.rope_cos, module.rope_sin
                    grown.append(
                        (module, module.max_position_embeddings, old_cos, old_sin)
                    )
                    module.max_position_embeddings = length
                    if callable(reset):
                        reset()
                    elif callable(build):
                        build(positions.device)
                    if old_cos is not None:
                        # Rebuilding on another device may differ by float
                        # rounding; reject only formula changes, then keep the
                        # original prefix bitwise.
                        new_cos = module.rope_cos[: len(old_cos)]
                        new_sin = module.rope_sin[: len(old_sin)]
                        if not (
                            torch.allclose(new_cos, old_cos.to(new_cos), atol=1e-4)
                            and torch.allclose(new_sin, old_sin.to(new_sin), atol=1e-4)
                        ):
                            raise ValueError(
                                "Native rotary cache growth changed existing positions"
                            )
                        new_cos.copy_(old_cos)
                        new_sin.copy_(old_sin)
        # Table-based HF ESM rotary modules do not consume position_ids.
        for module in rotary:
            saved.append(
                (module, module._seq_len_cached, module._cos_cached, module._sin_cached)
            )
            dummy = torch.empty(
                (1, 1, int(positions.max()) + 1, 1), device=positions.device
            )
            cos, sin = module._update_cos_sin_tables(dummy, seq_dimension=-2)
            module._cos_cached = cos[0, 0][positions].unsqueeze(1)
            module._sin_cached = sin[0, 0][positions].unsqueeze(1)
            module._seq_len_cached = positions.shape[1]
        if hasattr(model, "freqs_cis"):
            table = model.freqs_cis
            length = int(positions.max()) + 1
            if length > len(table):
                source = inspect.getmodule(type(model))
                if source is None or not hasattr(source, "precompute_freqs_cis"):
                    raise ValueError("Cannot grow native rotary cache")
                table = source.precompute_freqs_cis(
                    model.config.hidden_size // model.config.num_attention_heads, length
                ).to(table.device)
                if not torch.equal(model.freqs_cis, table[: len(model.freqs_cis)]):
                    raise ValueError(
                        "Native rotary cache growth changed existing positions"
                    )
            model.freqs_cis = table.to(positions.device)[positions[0]]
            yield
        else:
            gapped = not torch.equal(
                positions[0], torch.arange(positions.shape[1], device=positions.device)
            )
            if (
                gapped
                and not rotary
                and "position_ids" not in inspect.signature(model.forward).parameters
            ):
                raise ValueError("Backbone cannot consume original gapped positions")
            yield
    finally:
        if legacy_table is not None:
            model.freqs_cis = legacy_table
        for module, length, cos, sin in saved:
            module._seq_len_cached, module._cos_cached, module._sin_cached = (
                length,
                cos,
                sin,
            )
        for module, length, cos, sin in grown:
            module.max_position_embeddings = length
            module.rope_cos, module.rope_sin = cos, sin
