"""Build predictions for model-based inference ablations."""

from __future__ import annotations

import copy
import hashlib
import logging
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)


UNTRAINED_CONTROL = "untrained"
RANDOM_TRUNK_CONTROL = "random_trunk"
RANDOM_HEAD_CONTROL = "random_head"
RANDOM_BOTH_CONTROL = "random_both"
SCRAMBLED_LABELS_CONTROL = "scrambled_labels"
SCRAMBLED_SEQUENCES_CONTROL = "scrambled_sequences"
ControlCondition = Literal[
    "untrained",
    "random_trunk",
    "random_head",
    "random_both",
    "scrambled_labels",
    "scrambled_sequences",
]


RANDOMIZED_MODEL_CONTROLS = frozenset(
    {
        UNTRAINED_CONTROL,
        RANDOM_TRUNK_CONTROL,
        RANDOM_HEAD_CONTROL,
        RANDOM_BOTH_CONTROL,
    }
)
RANDOMIZED_TRUNK_CONTROLS = frozenset({RANDOM_TRUNK_CONTROL, RANDOM_BOTH_CONTROL})
RANDOMIZED_HEAD_CONTROLS = frozenset(
    {
        UNTRAINED_CONTROL,
        RANDOM_HEAD_CONTROL,
        RANDOM_BOTH_CONTROL,
    }
)
NO_INFERENCE_CONTROLS = frozenset({SCRAMBLED_LABELS_CONTROL})


def build_random_trunk(trunk: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Use identical random weights for development and test extraction."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_trunk = type(trunk)(copy.deepcopy(trunk.config))
    return random_trunk.to(next(trunk.parameters()).device).eval()


def contact_random_trunk_seed(seed: int, stream: str) -> int:
    """Derive stable, independent RNG streams for contact-trunk controls."""
    digest = hashlib.sha256(f"contact:random_trunk:{seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def build_control_model(
    control: str,
    trained_model: Any,
    seed: int,
    pretrained_model: Any = None,
) -> Any:

    # `from_pretrained`'s head init and `_init_weights` both draw from
    # torch's global RNG (there's no generator argument to inject), so
    # re-seed with `seed` here -- rather than once, up front, for the
    # whole run -- meaning each control's randomness depends only on
    # `seed`, not on the order controls happen to be built in.
    if control in RANDOMIZED_MODEL_CONTROLS:
        torch.manual_seed(seed)

    if control == UNTRAINED_CONTROL:
        if pretrained_model is None:
            raise ValueError("An untrained control requires a pretrained model.")
        model = pretrained_model
        model = model.to(next(trained_model.parameters()).device)
    elif control in RANDOMIZED_MODEL_CONTROLS - {UNTRAINED_CONTROL}:
        # Constructing from config lets each model class initialize all
        # parameters, including biases and normalization state that a
        # model-specific `_init_weights` may leave untouched.
        model = type(trained_model)(copy.deepcopy(trained_model.config)).to(
            next(trained_model.parameters()).device
        )
        base_prefix = model.base_model_prefix
        trained_state = trained_model.state_dict()
        if control == RANDOM_TRUNK_CONTROL:
            state_to_copy = {
                name: value
                for name, value in trained_state.items()
                if name != base_prefix and not name.startswith(f"{base_prefix}.")
            }
            model.load_state_dict(state_to_copy, strict=False)
        elif control == RANDOM_HEAD_CONTROL:
            state_to_copy = {
                name: value
                for name, value in trained_state.items()
                if name == base_prefix or name.startswith(f"{base_prefix}.")
            }
            model.load_state_dict(state_to_copy, strict=False)
    elif control == SCRAMBLED_SEQUENCES_CONTROL:
        # Same trained model; only the inputs are corrupted (see
        # `make_sequence_scrambler`), so no reinitialization is needed.
        model = trained_model
    else:
        raise ValueError(f"Unknown predict control condition: {control}")

    model.eval()
    return model
