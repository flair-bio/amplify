"""Optional FlashAttention 4 varlen kernel, wrapped as torch.compile-safe custom ops."""

import functools
import logging
from typing import Any, Protocol

import torch

try:
    # flash-attn-4 is an optional dependency of the project.
    from flash_attn.cute import interface as _fa4_interface
except ImportError:
    _fa4_interface = None


FA4VarlenReturnType = tuple[torch.Tensor, torch.Tensor]  # FA4 return type [out, lse]


class FlashAttn4VarlenFunc(Protocol):
    """Call signature of the FA4 varlen attention function, usable without FA4."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> FA4VarlenReturnType:
        """Attention over packed sequences delimited by ``cu_seqlens``."""
        ...


@torch.library.custom_op("flair::fa4_varlen_fwd", mutates_args=())
def _fa4_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> FA4VarlenReturnType:
    """FA4 varlen attention forward, as a custom op so torch.compile won't trace it.

    Gradients come from ``_fa4_varlen_bwd`` via ``register_autograd`` below.
    """
    return _fa4_interface.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        return_lse=True,
    )


@_fa4_varlen_fwd.register_fake
def _fa4_varlen_fwd_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *_: object
) -> FA4VarlenReturnType:
    """Shapes and dtypes of the forward's outputs, without running FA4.

    torch.compile traces with fake tensors and needs this to know what the op
    returns.
    """
    total_q, num_heads = q.shape[:2]
    out = q.new_empty((total_q, num_heads, v.shape[-1]))
    lse = q.new_empty((num_heads, total_q), dtype=torch.float32)
    return out, lse


@torch.library.custom_op("flair::fa4_varlen_bwd", mutates_args=())
def _fa4_varlen_bwd(
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FA4 varlen attention backward, wrapped as a custom op.

    Prevents torch.compile from attempting to trace the backward pass and
    trying to lower the CuTe launcher of the FA4 kernel.
    """
    dq, dk, dv = _fa4_interface._flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout.contiguous(),
        lse,
        causal=causal,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return dq, dk, dv


@_fa4_varlen_bwd.register_fake
def _fa4_varlen_bwd_fake(
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *_: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shapes and dtypes of the backward's outputs, for tracing with fake tensors."""
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _fa4_varlen_setup_context(
    ctx: Any, inputs: tuple[Any, ...], output: FA4VarlenReturnType
) -> None:
    """Save the forward's inputs and outputs for the backward pass."""
    # inputs: 5 tensors (q, k, v, cu_seqlens_q, cu_seqlens_k), then 3 scalars.
    ctx.save_for_backward(*output, *inputs[:5])
    ctx.scalars = inputs[5:]


def _fa4_varlen_backward(
    ctx: Any, dout: torch.Tensor, dlse: torch.Tensor
) -> tuple[torch.Tensor | None, ...]:
    """Gradient of `_fa4_varlen_fwd`. A custom op has none unless registered."""
    dq, dk, dv = _fa4_varlen_bwd(dout, *ctx.saved_tensors, *ctx.scalars)
    # No grads for cu_seqlens_q/k, max_seqlen_q/k, causal.
    return dq, dk, dv, None, None, None, None, None


_fa4_varlen_fwd.register_autograd(
    _fa4_varlen_backward, setup_context=_fa4_varlen_setup_context
)


def hardware_supports_fa4() -> bool:
    """True if the CUDA device is Hopper (sm90) or newer, as FA4 requires."""
    cc_hopper = (9, 0)  # Represents SM90 - blackwell is SM100 or (10, 0)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= cc_hopper


@functools.cache
def get_fa4_varlen_func() -> FlashAttn4VarlenFunc | None:
    """
    Returns the FA4's varlen attention kernel, if it's available and supported.
    """
    _logger = logging.getLogger(name=__name__)

    if not hardware_supports_fa4():
        _logger.info("Using PyTorch variable-length attention backend.")
        return None

    if _fa4_interface is None:
        _logger.info(
            "FlashAttention 4 is not installed on capable hardware; "
            "using PyTorch variable-length attention.",
        )
        return None

    _logger.info("Using FlashAttention 4 variable-length attention backend.")
    return _fa4_varlen_fwd
