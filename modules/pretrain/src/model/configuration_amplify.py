"""Hugging Face config class for AMPLIFY."""

from typing import Any, Literal

from transformers import PretrainedConfig


class AMPLIFYConfig(PretrainedConfig):
    """Configuration for the AMPLIFY protein language model.

    Attributes:
        hidden_size: Dimension of hidden representations.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        qk_norm: Apply per-head normalization to queries and keys before RoPE.
        attention_gate: Apply elementwise sigmoid gating before the attention output projection.
        nonlinear_mlm_head: Add a dense-GELU-norm MLM head transform.
        layernorm_scaling: Scale block pre-norm outputs by inverse square root of depth.
        intermediate_size: Dimension of the SwiGLU feed-forward inner layer.
        gaussian_init: Use zero-mean Gaussian initialization for linear and embedding weights.
        embedding_init_range: Uniform bound or Gaussian standard deviation for token embeddings.
        decoder_init_range: Uniform bound or Gaussian standard deviation for linear weights.
        norm_eps: Epsilon for the configured normalization layers.
        norm_type: Normalization layer type: ``"RMSNorm"``, ``"CenteredRMSNorm"``,
            ``"ZeroCenteredRMSNorm"``, or ``"LayerNorm"``. Also applies to enabled
            QK normalization and the nonlinear MLM head.
        vocab_size: Number of tokens in the amino-acid vocabulary.
        max_position_embeddings: Maximum context length supported by RoPE.
        rope_theta: Base frequency for RoPE.
    """

    model_type = "amplify"

    def __init__(
        self,
        hidden_size: int = 960,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 15,
        intermediate_size: int = 2560,
        embedding_init_range: float = 0.02,
        decoder_init_range: float = 0.02,
        norm_eps: float = 1e-05,
        vocab_size: int = 32,
        pad_token_id: int = 0,
        bos_token_id: int = 3,
        eos_token_id: int = 4,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        qk_norm: bool = False,
        attention_gate: bool = False,
        nonlinear_mlm_head: bool = False,
        layernorm_scaling: bool = False,
        norm_type: Literal[
            "RMSNorm", "CenteredRMSNorm", "ZeroCenteredRMSNorm", "LayerNorm"
        ] = "RMSNorm",
        gaussian_init: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.qk_norm = qk_norm
        self.attention_gate = attention_gate
        self.nonlinear_mlm_head = nonlinear_mlm_head
        self.layernorm_scaling = layernorm_scaling
        self.intermediate_size = intermediate_size
        self.gaussian_init = gaussian_init
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        if norm_type not in {
            "RMSNorm",
            "CenteredRMSNorm",
            "ZeroCenteredRMSNorm",
            "LayerNorm",
        }:
            raise ValueError(
                f"Unknown norm_type {norm_type!r}; expected 'RMSNorm', "
                "'CenteredRMSNorm', 'ZeroCenteredRMSNorm', or 'LayerNorm'."
            )
        self.norm_type = norm_type
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
