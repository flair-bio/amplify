"""Native checkpoint loading and training-only alignment projections."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from modules.post_train.src.contracts.config import AlignmentConfig
from modules.post_train.src.dataset.collator import Batch
from modules.post_train.src.model.tokenization import Vocabulary

TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json")


def native_assets(checkpoint: Path) -> tuple[str, ...]:
    """Find checkpoint-owned code through HF metadata, independent of layout."""
    from transformers.dynamic_module_utils import get_relative_import_files

    config = json.loads((checkpoint / "config.json").read_text())
    auto_map = config.get("auto_map", {})
    if (
        config.get("model_type") != "amplify"
        or not isinstance(auto_map, dict)
        or not isinstance(auto_map.get("AutoConfig"), str)
        or not isinstance(auto_map.get("AutoModelForMaskedLM"), str)
    ):
        raise ValueError("Require self-contained native AMPLIFY auto_map assets.")
    for name in TOKENIZER_ASSETS:
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing native tokenizer asset: {name}")
    tokenizer_config = json.loads((checkpoint / "tokenizer_config.json").read_text())
    tokenizer_map = tokenizer_config.get("auto_map", {})
    if not isinstance(tokenizer_map, dict):
        raise ValueError("Invalid tokenizer auto_map.")
    assets = set(TOKENIZER_ASSETS)
    for reference in (*auto_map.values(), *tokenizer_map.values()):
        references = reference if isinstance(reference, (list, tuple)) else [reference]
        for target in references:
            if target is None:
                continue
            if not isinstance(target, str):
                raise ValueError("Invalid native auto_map reference.")
            parts = target.split(".")
            if len(parts) != 2 or not all(part.isidentifier() for part in parts):
                raise ValueError(
                    f"Require local module.Class auto_map references: {target!r}"
                )
            source = checkpoint / f"{parts[0]}.py"
            if not source.is_file():
                raise ValueError(f"Missing native source asset: {source.name}")
            try:
                dependencies = [source, *map(Path, get_relative_import_files(source))]
            except FileNotFoundError as error:
                raise ValueError(
                    f"Missing native source dependency: {error.filename}"
                ) from error
            for dependency in dependencies:
                relative = dependency.relative_to(checkpoint)
                if ".." in relative.parts or relative.suffix != ".py":
                    raise ValueError(
                        "Native source dependencies must stay within the checkpoint."
                    )
                assets.add(str(relative))
    for name in ("special_tokens_map.json", "added_tokens.json"):
        if (checkpoint / name).is_file():
            assets.add(name)
    return tuple(sorted(assets))


def load_student(
    checkpoint: Path,
    device: torch.device,
    *,
    trust_remote_code: bool = False,
    required_positions: int,
    vocabulary: Vocabulary,
) -> Any:
    """Load complete checkpoint-owned code; never substitute a decoder/model."""
    from transformers import AutoModelForMaskedLM

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not trust_remote_code:
        raise ValueError(
            "Review native checkpoint Python, then set student.trust_remote_code=true."
        )
    native_assets(checkpoint)
    native, loading = AutoModelForMaskedLM.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Native AMPLIFY checkpoint did not load strictly: {loading}")
    if (
        not hasattr(native, "amplify")
        or not hasattr(native.amplify, "norm")
        or not hasattr(native, "lm_head")
        or native.config.vocab_size != vocabulary.size
        or getattr(native.config, "dual_mlm_head", False)
        or native.config.max_position_embeddings < required_positions
        or (
            getattr(native.config, "mask_token_id", None) is not None
            and native.config.mask_token_id != vocabulary.mask_id
        )
        or (
            native.config.pad_token_id,
            native.config.bos_token_id,
            native.config.eos_token_id,
        )
        != (vocabulary.pad_id, vocabulary.bos_id, vocabulary.eos_id)
    ):
        raise ValueError(
            "Unsupported native AMPLIFY single-MLM-head interface or context capacity."
        )
    return native.to(device)


class AlignmentModel(nn.Module):
    def __init__(
        self,
        student: Any,
        alignment: AlignmentConfig | None = None,
        *,
        teacher_dim: int,
    ) -> None:
        super().__init__()
        self.alignment = alignment if alignment is not None else AlignmentConfig()
        self.student = student
        self.sequence_projection = nn.Linear(
            student.config.hidden_size, self.alignment.projection_dim
        )
        self.structure_projection = nn.Linear(
            teacher_dim, self.alignment.projection_dim
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(-math.log(self.alignment.initial_temperature))
        )

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        masked = self.student(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        captured = []

        def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured.append(inputs[0])

        hook = self.student.amplify.norm.register_forward_pre_hook(capture)
        try:
            self.student(
                input_ids=batch["clean_input_ids"],
                attention_mask=batch["attention_mask"],
            )
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one native final normalization call.")
        residues = captured[0][batch["residue_mask"]]
        teacher = batch["embeddings"].detach()
        if residues.shape[0] != teacher.shape[0]:
            raise ValueError("Sequence and teacher residue counts differ.")
        return {
            "mlm_logits": masked.logits,
            "sequence": F.normalize(self.sequence_projection(residues), dim=-1),
            "structure": F.normalize(self.structure_projection(teacher), dim=-1),
        }


def forward(model: AlignmentModel, batch: Batch) -> dict[str, torch.Tensor]:
    device = batch["input_ids"].device
    # New context on each call prevents detached autocast weights leaking from
    # the no-grad cache pass into replay.
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        return model(batch)


def export_student(model: AlignmentModel, checkpoint: Path, output: Path) -> Path:
    destination = output / "amplify"
    model.student.save_pretrained(destination, safe_serialization=True)
    for name in native_assets(checkpoint):
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / name, destination / name)
    torch.save(
        {
            "sequence_projection": {
                key: value.detach().cpu()
                for key, value in model.sequence_projection.state_dict().items()
            },
            "structure_projection": {
                key: value.detach().cpu()
                for key, value in model.structure_projection.state_dict().items()
            },
            "logit_scale": model.logit_scale.detach().cpu(),
        },
        output / "alignment_heads.pt",
    )
    return destination
