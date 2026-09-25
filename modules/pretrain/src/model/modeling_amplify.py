import functools
import logging
import math
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.attention.varlen import varlen_attn
from torch.nn.functional import scaled_dot_product_attention, cross_entropy
from pydantic import BaseModel, ConfigDict, Field

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from .configuration_amplify import AMPLIFYConfig

_HALF_DTYPES = (torch.bfloat16, torch.float16)
_logger = logging.getLogger(__name__)


def _supports_fa4_hardware() -> bool:
    """Return whether the active CUDA device has FA4 kernels - Hopper (sm90) and greater."""
    if not torch.cuda.is_available():
        return False

    cc_major, _ = torch.cuda.get_device_capability()
    return cc_major >= 9


class FlashAttn4VarlenFunc(Protocol):
    """Callable signature of flash_attn.cute.flash_attn_varlen_func"""

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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@functools.cache
def _get_fa4_varlen_func() -> FlashAttn4VarlenFunc | None:
    """Load the optional FlashAttention 4 varlen kernel, if installed and supported."""
    if not _supports_fa4_hardware():
        _logger.info("Using PyTorch variable-length attention backend.")
        return None

    try:
        from flash_attn.cute import flash_attn_varlen_func

        _logger.info("Using FlashAttention 4 variable-length attention backend.")
        return flash_attn_varlen_func
    except ImportError as error:
        _logger.info(
            "FlashAttention 4 is not installed on capable hardware; "
            "using PyTorch variable-length attention.",
            exc_info=error,
        )
        return None


class AMPLIFYModelConfig(BaseModel):
    """Pydantic config for the AMPLIFY model architecture.

    Integrates with the OmegaConf + Pydantic config system used across this
    codebase.  Use :func:`~modules.pretrain.src.amplify.model.utils.config_loader.load_and_parse`
    to build an instance from YAML files and optional CLI dotlist overrides,
    then call :func:`get_amplify_model` (or one of the ``get_amplify_*``
    variants) to instantiate the model.

    Example of usage:
        cfg = load_and_parse("model.yaml", AMPLIFYModelConfig)
        model = get_amplify_masked_lm(cfg)

    Attributes:
        hidden_size: Dimension of hidden representations.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Dimension of the SwiGLU feed-forward inner layer.
        embedding_init_range: Uniform init bound for the token embedding.
        decoder_init_range: Uniform init bound for linear layers.
        norm_eps: Epsilon for RMSNorm layers.
        vocab_size: Number of tokens in the amino-acid vocabulary.
        pad_token_id: Index of the padding token.
        bos_token_id: Index of the beginning-of-sequence token.
        eos_token_id: Index of the end-of-sequence token.
        max_position_embeddings: Maximum context length supported by RoPE.
        rope_theta: Base frequency for RoPE.
        num_labels: Number of output labels for classification heads.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(960, gt=0)
    num_hidden_layers: int = Field(32, gt=0)
    num_attention_heads: int = Field(15, gt=0)
    intermediate_size: int = Field(2560, gt=0)
    embedding_init_range: float = Field(0.02, gt=0.0)
    decoder_init_range: float = Field(0.02, gt=0.0)
    norm_eps: float = Field(1e-5, gt=0.0)
    vocab_size: int = Field(32, gt=0)
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 4
    max_position_embeddings: int = Field(2048, gt=0)
    rope_theta: float = Field(10000.0, gt=0.0)
    num_labels: int = Field(2, gt=0)

    def to_hf_config(self) -> "AMPLIFYConfig":
        """Return an `AMPLIFYConfig` HuggingFace class from this config.

        ``num_labels`` is forwarded via ``**kwargs`` so that classification
        heads can read ``config.num_labels`` as expected.
        """
        data = self.model_dump()
        num_labels = data.pop("num_labels")
        return AMPLIFYConfig(**data, num_labels=num_labels)


class RMSNorm(nn.Module):
    """RMSNorm with manual weight casting to work around autocast bug.

    See https://github.com/pytorch/pytorch/issues/167308
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.rms_norm(
            x, self.weight.shape, self.weight.to(x.dtype), self.eps
        )


class RotaryEmbedding(nn.Module):
    """Rotary position embedding following LLaMA.

    Pre-computes cos/sin tables for all positions up to
    ``max_position_embeddings``, so the forward pass is an index lookup.
    """

    def __init__(self, config: "AMPLIFYConfig") -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self._build_tables(torch.device("cpu"))

    def _build_tables(self, device: torch.device) -> None:
        inv_freq = 1.0 / self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim
        )
        freqs = torch.outer(
            torch.arange(self.max_position_embeddings, device=device).float(), inv_freq
        )
        self.rope_cos = freqs.cos()
        self.rope_sin = freqs.sin()

    def reset_tables(self) -> None:
        """Recompute the cos/sin tables in place, preserving the current device."""
        device = (
            self.rope_cos.device if self.rope_cos is not None else torch.device("cpu")
        )
        self._build_tables(device)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rebuild if unbuilt or moved device. Introduces a graph break.
        if self.rope_cos is None or self.rope_cos.device != x.device:
            self._build_tables(x.device)
        return self.rope_cos[position_ids], self.rope_sin[position_ids]


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply LLaMA-style rotary embeddings to *x* (batch, seq, heads, dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RotaryQKV(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def apply_rotary_emb_qkv(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> RotaryQKV:
    """RoPE for the packed path: returns rotated q/k and v as-is."""
    q, k, v = qkv.unbind(1)
    cos, sin = cos.unsqueeze(-2).to(q.dtype), sin.unsqueeze(-2).to(q.dtype)
    return RotaryQKV(
        q=apply_rotary_emb(q, cos, sin), k=apply_rotary_emb(k, cos, sin), v=v
    )


class Attention(nn.Module):
    """MHSA with RoPE.

    Supports padded batches (PyTorch SDPA) and packed sequences (FlashAttention
    when optionally installed, otherwise PyTorch variable-length attention).
    Receives the pre-norm layer from the caller, applies it internally, and returns
    ``hidden_states`` with the attention residual already added.
    """

    def __init__(self, config: AMPLIFYConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.fa4_attn_fn = _get_fa4_varlen_func()

        # Multi-head attention projections (query, key, value combined + output).
        self.qkv_proj = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias=False
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    # Fast path: packed sequences with optional FlashAttention acceleration.
    def _attn_packed(
        self,
        layernorm: nn.Module,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int],
    ) -> torch.Tensor:
        """Variable-length attention path for packed sequences."""
        assert max_seqlen is not None, (
            "max_seqlen must be provided for packed sequences"
        )
        batch_size, total_tokens = hidden_states.shape[:2]
        hidden_states_flat = hidden_states.view(-1, self.num_heads * self.head_dim)
        # Pre-norm + QKV projection + RoPE.
        normed = layernorm(hidden_states_flat)
        qkv = self.qkv_proj(normed).view(total_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = apply_rotary_emb_qkv(qkv, cos, sin)
        if q.dtype not in _HALF_DTYPES:
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        if self.fa4_attn_fn is None:
            attn_output = varlen_attn(
                q,
                k,
                v,
                cu_seq_q=cu_seqlens,
                cu_seq_k=cu_seqlens,
                max_q=max_seqlen,
                max_k=max_seqlen,
            )
        else:
            attn_output, _ = self.fa4_attn_fn(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
            )
        attn_output = attn_output.to(hidden_states_flat.dtype).view(
            -1, self.num_heads * self.head_dim
        )
        # Fused attention residual: hidden + attn_output @ o_proj.weight.T
        hidden_states_flat = torch.addmm(
            hidden_states_flat, attn_output, self.o_proj.weight.t()
        )
        return hidden_states_flat.view(batch_size, total_tokens, -1)

    # Standard path: padded batches with PyTorch SDPA.
    def _attn_padded(
        self,
        layernorm: nn.Module,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """PyTorch SDPA path for padded batches."""
        batch_size, seq_len = hidden_states.shape[:2]
        # Pre-norm before attention.
        normed = layernorm(hidden_states)
        # Project to Q, K, V and reshape to (batch, seq, heads, dim).
        qkv = self.qkv_proj(normed).view(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(2)
        # Apply rotary position embeddings, then transpose to (batch, heads, seq, dim).
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        if output_attentions:
            # Manual attention computation to return attention weights.
            attn_weights = (q @ k.transpose(-2, -1)) * (q.size(-1) ** -0.5)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
            attn_output = (
                (attn_weights.softmax(-1) @ v)
                .transpose(1, 2)
                .reshape(batch_size, seq_len, -1)
            )
        else:
            # Efficient fused attention (does not return attention weights).
            attn_output = scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            attn_weights = None

        return hidden_states + self.o_proj(attn_output), attn_weights

    def forward(
        self,
        layernorm: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calls the correct path (packed or padded)."""
        # packed path
        if cu_seqlens is not None:
            return self._attn_packed(
                layernorm, hidden_states, cos, sin, cu_seqlens, max_seqlen
            ), None
        # padded path
        return self._attn_padded(
            layernorm, hidden_states, cos, sin, attention_mask, output_attentions
        )


class SwiGLU(nn.Module):
    """SwiGLU FFN: gate + up projection, SiLU, then down projection.

    Receives the pre-norm layer from the caller, applies it internally, and returns
    ``hidden_states`` with the FFN residual added.
    """

    def __init__(self, config: AMPLIFYConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    # Packed path for SwiGLU
    def _mlp_packed(
        self, layernorm: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Packed path using the same "silu(gate) * up" as the padded path below."""
        batch_size, total_tokens = hidden_states.shape[:2]
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        normed = layernorm(hidden_states_flat)
        gate, up = self.gate_up_proj(normed).chunk(2, dim=-1)
        swiglu_out = self.act_fn(gate) * up
        # Fused residual: hidden + swiglu(gate, up) @ down_proj.weight.T
        hidden_states_flat = torch.addmm(
            hidden_states_flat, swiglu_out, self.down_proj.weight.t()
        )
        # Restore 3D for the next layer or the final LM head.
        return hidden_states_flat.view(batch_size, total_tokens, -1)

    # Padded path for SwiGLU
    def _mlp_padded(
        self, layernorm: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Padded path using standard SiLU gating."""
        # Pre-norm + SwiGLU FFN + residual connection.
        gate, up = self.gate_up_proj(layernorm(hidden_states)).chunk(2, dim=-1)
        return hidden_states + self.down_proj(self.act_fn(gate) * up)

    def forward(
        self,
        layernorm: nn.Module,
        hidden_states: torch.Tensor,
        packed: bool = False,
    ) -> torch.Tensor:
        """Calls the correct SwiGLU path (packed or padded)."""
        if packed:
            return self._mlp_packed(layernorm, hidden_states)
        return self._mlp_padded(layernorm, hidden_states)


class EncoderBlock(nn.Module):
    """Transformer encoder block with RoPE attention and SwiGLU FFN."""

    # Maps old flat param. names to keep existing checkpoints loadable.
    _OLD_KEY_MAP = {
        "qkv_proj.weight": "attention.qkv_proj.weight",
        "o_proj.weight": "attention.o_proj.weight",
        "gate_up_proj.weight": "mlp.gate_up_proj.weight",
        "down_proj.weight": "mlp.down_proj.weight",
    }

    def __init__(self, config: AMPLIFYConfig) -> None:
        super().__init__()
        # Pre-normalization layer applied before attention.
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        # Attention module.
        self.attention = Attention(config)
        # Pre-normalization layer applied before FFN.
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        # FFN module.
        self.mlp = SwiGLU(config)

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        # Remap old flat keys to their new nested locations.
        for old_suffix, new_suffix in self._OLD_KEY_MAP.items():
            old_key, new_key = prefix + old_suffix, prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        output_attentions: bool,
        max_seqlen: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pre-norm attention + SwiGLU FFN with residual connections.

        Returns (hidden_states, attn_weights).
        """
        packed = cu_seqlens is not None

        # Attention block
        hidden_states, attn_weights = self.attention(
            self.input_layernorm,
            hidden_states,
            attention_mask,
            cos,
            sin,
            output_attentions,
            max_seqlen,
            cu_seqlens,
        )
        # SwiGLU FFN block
        hidden_states = self.mlp(
            self.post_attention_layernorm, hidden_states, packed=packed
        )
        return hidden_states, attn_weights


class AMPLIFYPreTrainedModel(PreTrainedModel):
    """Hugging Face base class with AMPLIFY-specific parameter initialization."""

    config_class = AMPLIFYConfig
    base_model_prefix = "amplify"

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize linear layers and embeddings with uniform weights."""
        if isinstance(module, nn.Linear):
            nn.init.uniform_(
                module.weight,
                -self.config.decoder_init_range,
                self.config.decoder_init_range,
            )
            if module.bias is not None:
                # `from_pretrained` fills checkpoint-missing params via this hook
                # only; skipping bias left task heads on uninitialised memory.
                fan_in = module.weight.shape[1]
                bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(
                module.weight,
                -self.config.embedding_init_range,
                self.config.embedding_init_range,
            )
        elif isinstance(module, RotaryEmbedding):
            # RoPE tables are non-persistent buffers, so they are absent from the
            # checkpoint and `from_pretrained` reallocates them as uninitialised
            # memory. The guard in `forward` cannot detect that (right shape and
            # device), so without this a reloaded checkpoint returns NaN.
            module.reset_tables()


class AMPLIFYModel(AMPLIFYPreTrainedModel):
    """AMPLIFY base encoder without any task head.

    Returns contextual embeddings from the transformer encoder stack.
    Used for embedding extraction, the most common downstream use case for
    protein language models.

    Supports both padded batches (PyTorch SDPA) and packed sequences
    (PyTorch variable-length attention).
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.post_init()

    @staticmethod
    def _validate_packed_inputs(
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> None:
        """Static helper to validate packed inputs."""
        if position_ids is None:
            raise ValueError(
                "Packed sequences require explicit position_ids derived from cu_seqlens."
            )
        if output_attentions:
            raise ValueError(
                "Packed sequences do not support returning attention weights."
            )
        if not hidden_states.is_cuda:
            raise ValueError("Packed sequences require CUDA.")

    @staticmethod
    def _prepare_position_ids(
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        packed: bool,
    ) -> torch.Tensor:
        """Static helper to prepare position ids based on mode (packed or standard)."""
        if packed:
            # packed mode: squeeze position ids
            assert position_ids is not None
            return position_ids.to(device=input_ids.device, dtype=torch.long).squeeze(0)
        # standard mode: build position ids
        if position_ids is not None:
            return position_ids.to(device=input_ids.device, dtype=torch.long).expand(
                input_ids.shape[0], -1
            )
        return (
            torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Static helper to prepare attention mask."""
        if attention_mask is None:
            return None
        return attention_mask.unsqueeze(1).unsqueeze(1).bool()

    def _compute_rope(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        packed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Static helper to compute rotary position embeddings."""
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if not packed:
            # Standard path needs a head broadcast dimension: (B, S, D/2) -> (B, S, 1, D/2).
            cos = cos.unsqueeze(2).to(hidden_states.dtype)
            sin = sin.unsqueeze(2).to(hidden_states.dtype)
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutput:
        """Run the encoder stack and return contextual embeddings.

        Args:
            input_ids: Token IDs — ``(B, S)`` padded or ``(1, T)`` packed.
            position_ids: Position IDs — same shape as ``input_ids``.
            attention_mask: ``(B, S)`` boolean mask for padded mode.
            cu_seqlens: Cumulative sequence lengths for packed mode.
            max_seqlen: Maximum sequence length in the packed batch.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights (padded mode only).

        Returns:
            :class:`BaseModelOutput` with ``last_hidden_state``, optional
            ``hidden_states``, and optional ``attentions``.
        """
        all_hidden_states: Optional[List[torch.Tensor]] = (
            [] if output_hidden_states else None
        )
        all_attentions: Optional[List[torch.Tensor]] = [] if output_attentions else None
        hidden_states = self.embed_tokens(input_ids)

        packed = cu_seqlens is not None

        # Packed mode: validate requirements.
        if packed:
            self._validate_packed_inputs(hidden_states, position_ids, output_attentions)
        # Standard mode: reshape attention mask.
        else:
            attention_mask = self._prepare_attention_mask(attention_mask)
        # Prepare position ids depending on mode.
        position_ids = self._prepare_position_ids(input_ids, position_ids, packed)
        # Compute RoPE depending on mode.
        cos, sin = self._compute_rope(hidden_states, position_ids, packed)

        # Run through all encoder layers.
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask,
                cos,
                sin,
                output_attentions,
                max_seqlen,
                cu_seqlens,
            )

            if all_attentions is not None:
                all_attentions.append(attn_weights)

        # Final layer norm.
        hidden_states = self.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class AMPLIFYForMaskedLM(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a masked language model head.

    Used for pretraining (MLM), pseudo-perplexity scoring, zero-shot variant
    effect prediction, and MLM fine-tuning.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.amplify = AMPLIFYModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> MaskedLMOutput:
        """Forward pass for masked language modeling.

        Args:
            input_ids: Token IDs — ``(B, S)`` padded or ``(1, T)`` packed.
            labels: MLM labels with ``-100`` for unmasked positions.
            position_ids: Position IDs — same shape as ``input_ids``.
            attention_mask: ``(B, S)`` boolean mask for padded mode.
            cu_seqlens: Cumulative sequence lengths for packed mode.
            max_seqlen: Maximum sequence length in the packed batch.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights (padded mode only).

        Returns:
            :class:`MaskedLMOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.lm_head(encoder_output.last_hidden_state)

        loss = None
        if labels is not None:
            loss = cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class AMPLIFYForSequenceClassification(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a sequence-level classification head.

    Mean-pools over non-padding tokens, then projects to ``num_labels``.
    Used for protein family classification, localization, solubility prediction.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.amplify = AMPLIFYModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        num_sequences: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        """Forward pass for sequence classification with padded or packed input.

        Args:
            input_ids: Token IDs ``(B, S)``.
            labels: Class labels ``(B,)`` for cross-entropy or ``(B, num_labels)`` for regression.
            position_ids: Position IDs ``(B, S)``.
            attention_mask: Boolean mask ``(B, S)``, ``True`` for real tokens.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights.

        Returns:
            :class:`SequenceClassifierOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Mean-pool over non-padding tokens.
        hidden_states = encoder_output.last_hidden_state
        if cu_seqlens is not None:
            if num_sequences is None:
                raise ValueError(
                    "Packed sequence classification requires num_sequences."
                )
            lengths = (
                cu_seqlens[1 : num_sequences + 1] - cu_seqlens[:num_sequences]
            ).long()
            segments = torch.repeat_interleave(
                torch.arange(num_sequences, device=hidden_states.device), lengths
            )
            pooled = hidden_states.new_zeros(num_sequences, hidden_states.shape[-1])
            pooled.index_add_(0, segments, hidden_states[0, : segments.numel()])
            pooled = pooled / lengths.clamp(min=1).unsqueeze(-1)
        elif attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.functional.mse_loss(
                    logits.squeeze(-1), labels.to(logits.dtype)
                )
            else:
                loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


class AMPLIFYForTokenClassification(AMPLIFYPreTrainedModel):
    """AMPLIFY encoder with a per-token classification head.

    Projects each token representation to ``num_labels``.
    Used for binding site prediction, secondary structure, PTM site prediction.
    """

    def __init__(self, config: AMPLIFYConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.amplify = AMPLIFYModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        """Forward pass for token classification with padded or packed input.

        Args:
            input_ids: Token IDs ``(B, S)``.
            labels: Per-token labels ``(B, S)`` with ``-100`` for ignored positions.
            position_ids: Position IDs ``(B, S)``.
            attention_mask: Boolean mask ``(B, S)``.
            output_hidden_states: Return all intermediate hidden states.
            output_attentions: Return attention weights.

        Returns:
            :class:`TokenClassifierOutput` with loss, logits, hidden_states, attentions.
        """
        encoder_output = self.amplify(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = self.classifier(encoder_output.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
        )


_AUTO_MODEL_CLASSES: Dict[str, Type[PreTrainedModel]] = {
    "AutoModel": AMPLIFYModel,
    "AutoModelForMaskedLM": AMPLIFYForMaskedLM,
    "AutoModelForSequenceClassification": AMPLIFYForSequenceClassification,
    "AutoModelForTokenClassification": AMPLIFYForTokenClassification,
}


def _auto_map_target(cls: Type[Union[PreTrainedModel, PretrainedConfig]]) -> str:
    """Formats a class as the ``<module_basename>.<ClassName>`` string HF expects.

    Mirrors what `transformers.dynamic_module_utils.custom_object_save` computes,
    which is also the name the module file is copied under inside the checkpoint.
    """
    return f"{cls.__module__.rsplit('.', 1)[-1]}.{cls.__name__}"


def register_amplify_auto_classes() -> Dict[str, str]:
    """Registers AMPLIFY with the HF auto-classes and returns the ``auto_map``.

    `save_pretrained` only copies the defining source into a checkpoint (and
    records an `auto_map` entry) for registered classes; without this a
    checkpoint is not self-contained and `from_pretrained(...,
    trust_remote_code=True)` fails. All heads are registered, not just the one
    being saved, since the copied module defines them all and a checkpoint
    should be loadable as any AMPLIFY head.
    """
    # The config registers separately: its `register_for_auto_class` defaults to
    # "AutoConfig" and takes no argument, whereas each head must name its class.
    AMPLIFYConfig.register_for_auto_class()
    auto_map = {"AutoConfig": _auto_map_target(AMPLIFYConfig)}
    for auto_class, model_cls in _AUTO_MODEL_CLASSES.items():
        model_cls.register_for_auto_class(auto_class)
        auto_map[auto_class] = _auto_map_target(model_cls)
    return auto_map


def get_amplify_model(config: AMPLIFYModelConfig) -> AMPLIFYModel:
    """Create the base AMPLIFY encoder (no task head)."""
    return AMPLIFYModel(config.to_hf_config())


def get_amplify_masked_lm(config: AMPLIFYModelConfig) -> AMPLIFYForMaskedLM:
    """Create AMPLIFY with a MLM head (for pretraining)."""
    return AMPLIFYForMaskedLM(config.to_hf_config())


def get_amplify_sequence_classifier(
    config: AMPLIFYModelConfig,
) -> AMPLIFYForSequenceClassification:
    """Create AMPLIFY with a sequence-level classification head."""
    return AMPLIFYForSequenceClassification(config.to_hf_config())


def get_amplify_token_classifier(
    config: AMPLIFYModelConfig,
) -> AMPLIFYForTokenClassification:
    """Create AMPLIFY with a per-token classification head."""
    return AMPLIFYForTokenClassification(config.to_hf_config())
