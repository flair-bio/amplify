import logging
import torch
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Literal


class OptimizerConfig(BaseModel):
    """Strict Pydantic configuration matching the YAML fields.

    extra="forbid" ensures typos in the YAML (e.g., "learning_rate" instead of "lr")
    cause a validation error rather than silently failing.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["AdamW", "Adam", "Adafactor", "Lamb"] = Field(
        description="Optimizer type."
    )
    lr: float = Field(description="Learning rate.", gt=0.0)
    betas: tuple[float, float] = Field(
        description="Beta parameters for Adam-based optimizers."
    )
    eps: float | tuple[float, float] = Field(
        description="Epsilon for numerical stability. Adafactor expects a tuple."
    )
    weight_decay: float = Field(description="Weight decay coefficient.", ge=0.0)
    fused: bool = Field(
        description="Whether to use the fused optimizer implementation (faster on GPU)."
    )

    def sanitize_opt_kwargs(self) -> dict[str, Any]:
        """Build optimizer kwargs by removing non-constructor fields and handling incompatibilities."""
        kwargs = self.model_dump(
            exclude={"type", "weight_decay"},
            exclude_none=True,
        )

        # Fused kernels are only valid for Adam/AdamW.
        if self.type not in {"AdamW", "Adam"}:
            kwargs.pop("fused", None)

        if self.type == "Adafactor":
            # HF Adafactor does not accept standard Adam betas.
            kwargs.pop("betas", None)

            # If the user passed a standard float eps (likely intended for AdamW),
            # pop it. This safely forces Adafactor to use its native default (1e-30, 1e-3).
            if isinstance(kwargs.get("eps"), float):
                kwargs.pop("eps")

        return kwargs


def get_optimizer(
    model: torch.nn.Module, config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Create an optimizer for the model with proper weight decay separation.

    Args:
        model: Model whose parameters will be optimized.
        config: Validated strict Pydantic configuration.

    Returns:
        Configured optimizer instance.

    Raises:
        ImportError: If an external optimizer library is missing.
        RuntimeError: If fused is requested without CUDA availability.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    # Separation logic:
    # 1. 1D tensors bypass weight decay (Biases, LayerNorm scales).
    # 2. String matching explicitly protects Embeddings (which are 2D but suffer from weight decay).
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        name_lower = name.lower()
        if (
            param.dim() < 2
            or "embed" in name_lower
            or "position_embeddings" in name_lower
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters: list[dict[str, Any]] = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt_kwargs = config.sanitize_opt_kwargs()

    # Hardware sanity check: Fail loudly instead of silently overriding.
    if opt_kwargs.get("fused") and not torch.cuda.is_available():
        raise RuntimeError(
            "Fused optimizer requested in configuration, but CUDA is not available. "
            "Set `fused: false` in your YAML to train on CPU."
        )

    match str(config.type):
        case "AdamW":
            # Standard optimizer for Transformer-based pLMs.
            # Cited heavily in ESM-1b (Rives et al., 2021) and ESM-2 (Lin et al., 2023).
            return torch.optim.AdamW(optimizer_grouped_parameters, **opt_kwargs)

        case "Adam":
            # Standard legacy Adam (without decoupled weight decay).
            return torch.optim.Adam(optimizer_grouped_parameters, **opt_kwargs)

        case "Adafactor":
            # Memory-efficient optimizer used in ProtTrans (Elnaggar et al., 2021).
            # Crucial for training massive models like ProtT5 within GPU VRAM limits.
            try:
                from transformers.optimization import Adafactor

                return Adafactor(optimizer_grouped_parameters, **opt_kwargs)
            except ImportError:
                raise ImportError("Adafactor requires the `transformers` library.")

        case "Lamb":
            # Layer-wise adaptive rate scaling for massive batch sizes.
            # Enables distributed pre-training across large clusters (You et al., 2019).
            try:
                from torch_optimizer import Lamb

                return Lamb(optimizer_grouped_parameters, **opt_kwargs)
            except ImportError:
                raise ImportError("Lamb requires the `torch_optimizer` package.")

        case other:
            raise ValueError(
                f"Unsupported optimizer type: {other}. "
                "Only ['AdamW', 'Adam', 'Adafactor', 'Lamb'] are supported."
            )
