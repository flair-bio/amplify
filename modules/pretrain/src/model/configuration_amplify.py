from typing import Any

from transformers import PretrainedConfig


class AMPLIFYConfig(PretrainedConfig):
    """Configuration for the AMPLIFY protein language model.

    Attributes:
        hidden_size: Dimension of hidden representations.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Dimension of the SwiGLU feed-forward inner layer.
        embedding_init_range: Uniform init bound for the token embedding.
        decoder_init_range: Uniform init bound for linear layers.
        norm_eps: Epsilon for RMSNorm layers.
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
        **kwargs: Any,
    ):
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
        self.intermediate_size = intermediate_size
        self.embedding_init_range = embedding_init_range
        self.decoder_init_range = decoder_init_range
        self.norm_eps = norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
