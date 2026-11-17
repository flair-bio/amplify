"""Model resources and primitives for inference ablations."""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from torch import nn

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


@dataclass
class ControlPredictors:
    """Independent control models shared across test views."""

    random_trunk_model: nn.Module | BaseEstimator | None = None
    random_trunk_seed: int | None = None
    independent_models: dict[str, nn.Module | BaseEstimator] = field(
        default_factory=dict
    )


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


def make_sequence_scrambler(
    tokenizer: Any, seed: int
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build a batch_transform that shuffles token order within each sequence
    (leaving padding/special-token positions, as well as each sequence's
    first amino acid, in place), destroying sequence/positional
    information while preserving token composition.

    Loops over rows in Python (batch_size is small relative to a training
    loop, so this isn't performance-critical) and permutes each row's
    shufflable positions in place via ``torch.randperm``, rather than a
    single vectorized argsort over the whole batch. Uses its own
    ``torch.Generator`` seeded from `seed` (rather than torch's global
    RNG) so the shuffling is fully deterministic and independent of any
    other randomness consumed elsewhere in the run.
    """
    special_ids_list = list(tokenizer.all_special_ids)
    state: dict[str, Any] = {"generator": None, "device": None}

    def _scramble(batch: dict[str, Any]) -> dict[str, Any]:
        batch = dict(batch)
        input_ids = batch["input_ids"]
        device = input_ids.device
        batch_size, _ = input_ids.shape
        if state["generator"] is None or state["device"] != device:
            state["generator"] = torch.Generator(device=device).manual_seed(seed)
            state["device"] = device
        generator = state["generator"]
        special_ids = torch.tensor(
            special_ids_list, device=device, dtype=input_ids.dtype
        )
        movable = ~torch.isin(input_ids, special_ids)
        scrambled = input_ids.clone()
        if "cu_seqlens" in batch:
            offsets = batch["cu_seqlens"][: batch["num_sequences"] + 1].tolist()
            spans = [(0, start, end) for start, end in zip(offsets[:-1], offsets[1:])]
        else:
            spans = [(row, 0, input_ids.shape[1]) for row in range(batch_size)]
        for row, start, end in spans:
            movable_idx = movable[row, start:end].nonzero(as_tuple=True)[0] + start
            if movable_idx.numel() <= 1:
                continue
            shufflable_idx = movable_idx[1:]
            perm = shufflable_idx[
                torch.randperm(
                    shufflable_idx.numel(), generator=generator, device=device
                )
            ]
            scrambled[row, shufflable_idx] = input_ids[row, perm]
        batch["input_ids"] = scrambled
        return batch

    return _scramble
