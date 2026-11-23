"""Fixed-protocol independent contact controls."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from modules.evaluate.src.artifacts.results import FitResult
from modules.evaluate.src.model.contact_attention import ContactAttentionModel
from modules.evaluate.src.model.contact_jacobian import ContactJacobianModel
from modules.evaluate.src.model.contact_model import ContactPredictionModel
from modules.evaluate.src.model.controls import (
    build_random_trunk,
    contact_random_trunk_seed,
)
from modules.evaluate.src.model.factory import Predictor


@dataclass
class ControlInputs:
    """An independently constructed contact control before feature preparation."""

    model: torch.nn.Module
    seed: int
    metadata: dict[str, Any]
    serialization_model: torch.nn.Module


def build_contact_controls(
    predictor: Predictor, controls: Sequence[str], *, seed: int, device: torch.device
) -> dict[str, ControlInputs]:
    if "random_trunk" not in controls:
        return {}
    model = predictor.model
    if not isinstance(
        model, (ContactPredictionModel, ContactAttentionModel, ContactJacobianModel)
    ):
        raise TypeError("Contact controls require a contact predictor.")
    trunk_seed = contact_random_trunk_seed(seed, "trunk")
    head_seed = contact_random_trunk_seed(seed, "head")
    trunk = build_random_trunk(model.trunk, trunk_seed)
    if isinstance(model, ContactJacobianModel):
        random_model = ContactJacobianModel(
            trunk, predictor.tokenizer, mutant_batch_size=model.mutant_batch_size
        ).to(device)
    else:
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(head_seed)
            random_model = type(model)(trunk).to(device)
    return {
        "random_trunk": ControlInputs(
            model=random_model,
            seed=head_seed,
            serialization_model=trunk
            if isinstance(model, ContactJacobianModel)
            else random_model,
            metadata={
                "control": "random_trunk",
                "protocol": "contact-random-trunk-v1",
                "trunk_initialization": "random_from_config",
                "trunk_seed": trunk_seed,
                "head_seed": None
                if isinstance(model, ContactJacobianModel)
                else head_seed,
            },
        )
    }


def contact_control_settings(
    inputs: ControlInputs, result: FitResult
) -> dict[str, Any]:
    return {
        **inputs.metadata,
        "development_features": result.input_identity.get("development_features")
        if isinstance(inputs.model, ContactPredictionModel)
        else None,
        "fit": result.checkpoint_settings,
    }
