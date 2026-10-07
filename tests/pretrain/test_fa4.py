import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from modules.pretrain.src.model import fa4


def test_fake_matches_kernel_layout():
    """torch.compile traces the ops through their fake impls, so the shapes and
    dtypes there must match what FA4 returns."""
    total, heads, head_dim = 6, 4, 16
    with FakeTensorMode():
        q = torch.empty(total, heads, head_dim, dtype=torch.bfloat16)
        cu_seqlens = torch.empty(3, dtype=torch.int32)
        out, lse = fa4._fa4_varlen_fwd(q, q, q, cu_seqlens, cu_seqlens, 4, 4, False)
        dq, dk, dv = fa4._fa4_varlen_bwd(
            out, out, lse, q, q, q, cu_seqlens, cu_seqlens, 4, 4, False
        )

    assert out.shape == (total, heads, head_dim) and out.dtype == torch.bfloat16
    assert lse.shape == (heads, total) and lse.dtype == torch.float32
    assert dq.shape == dk.shape == dv.shape == q.shape


def _fa4_available() -> bool:
    return fa4.hardware_supports_fa4() and fa4._fa4_interface


@pytest.mark.skipif(not _fa4_available(), reason="Needs FA4 on a Hopper+ GPU.")
class TestFA4CustomOp:
    """The custom ops must be numerically the same kernel as
    `flash_attn_varlen_func`, while staying opaque to torch.compile."""

    def _inputs(self, strided_qkv: bool):
        """With ``strided_qkv``, q/k/v are strided views of one ``(T, 3, H, D)``
        tensor, like `Attention._attn_packed`'s ``qkv.unbind(1)``."""
        torch.manual_seed(0)
        cu_seqlens = torch.tensor([0, 100, 350, 512], dtype=torch.int32, device="cuda")
        if strided_qkv:
            qkv = torch.randn(512, 3, 4, 64, device="cuda", dtype=torch.bfloat16)
            q, k, v = qkv.requires_grad_().unbind(1)
        else:
            q, k, v = (
                torch.randn(
                    512, 4, 64, device="cuda", dtype=torch.bfloat16
                ).requires_grad_()
                for _ in range(3)
            )
        dout = torch.randn(512, 4, 64, device="cuda", dtype=torch.bfloat16)
        kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=250,
            max_seqlen_k=250,
            causal=False,
        )
        return q, k, v, dout, kwargs

    @staticmethod
    def _fwd_bwd(fn, q, k, v, dout, kwargs):
        for t in (q, k, v):
            t.grad = None
        out, _ = fn(q, k, v, **kwargs)
        out.backward(dout)
        return [out.detach()] + [t.grad for t in (q, k, v)]

    def test_matches_flash_attn_varlen_func(self):
        from flash_attn.cute import flash_attn_varlen_func

        q, k, v, dout, kwargs = self._inputs(strided_qkv=False)
        expected = self._fwd_bwd(flash_attn_varlen_func, q, k, v, dout, kwargs)
        actual = self._fwd_bwd(fa4._fa4_varlen_fwd, q, k, v, dout, kwargs)

        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=0)

    def test_compiles_as_a_single_graph(self):
        q, k, v, dout, kwargs = self._inputs(strided_qkv=False)
        compiled = torch.compile(fa4._fa4_varlen_fwd, fullgraph=True, dynamic=True)
        expected = self._fwd_bwd(fa4._fa4_varlen_fwd, q, k, v, dout, kwargs)
        actual = self._fwd_bwd(compiled, q, k, v, dout, kwargs)

        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, atol=1e-2, rtol=0)

    @pytest.mark.parametrize("strided_qkv", [False, True])
    def test_opcheck(self, strided_qkv):
        """Schema (no mutation/aliasing), fake impls vs the real kernel's
        metadata (incl. strides), autograd registration, and AOT dispatch."""
        q, k, v, dout, kwargs = self._inputs(strided_qkv)
        torch.library.opcheck(fa4._fa4_varlen_fwd, (q, k, v), kwargs)

        out, lse = fa4._fa4_varlen_fwd(q, k, v, **kwargs)
        bwd_args = (dout, out, lse, q, k, v)
        torch.library.opcheck(
            fa4._fa4_varlen_bwd, tuple(t.detach() for t in bwd_args), kwargs
        )
