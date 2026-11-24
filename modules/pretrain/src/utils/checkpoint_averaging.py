"""Uniformly average HF checkpoints of one run into a single HF checkpoint.

Typically the last few exports of the LR decay (see `trainer.hf_save_at_steps`),
i.e. Polyak / LAWA averaging. The trainer does this at the end of training
when `trainer.average_hf_checkpoints` is set; otherwise, average a finished
run by hand. Runs on CPU, streaming one tensor at a time.

    python -m modules.pretrain.src.utils.checkpoint_averaging \\
        --checkpoints runs/x/hf_checkpoints/checkpoint_{98000,99000,100000} \\
        --output-dir runs/x/hf_checkpoints/averaged
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS_NAME = "model.safetensors"
MANIFEST_NAME = "merge_manifest.json"
# Written by `save_pretrained`; differ across exports without changing the model.
_CONFIG_METADATA = {"_name_or_path", "transformers_version"}


def _config(path: Path) -> dict:
    config = json.loads((path / "config.json").read_text())
    return {k: v for k, v in config.items() if k not in _CONFIG_METADATA}


def _tensor_specs(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    if (path / f"{WEIGHTS_NAME}.index.json").exists():
        raise ValueError(f"Sharded checkpoints are not supported: {path}")
    if not (path / WEIGHTS_NAME).is_file():
        raise FileNotFoundError(f"No {WEIGHTS_NAME} in {path}")
    with safe_open(path / WEIGHTS_NAME, framework="pt") as f:
        slices = {key: f.get_slice(key) for key in f.keys()}  # noqa: SIM118
        return {k: (tuple(s.get_shape()), s.get_dtype()) for k, s in slices.items()}


def average_checkpoints(checkpoints: Sequence[Path], output_dir: Path) -> Path:
    """Write the uniform average of `checkpoints` to `output_dir`.

    The checkpoints must share `config.json` and tensor keys, shapes and dtypes.
    Floating tensors are averaged in FP64 and cast back to their dtype (so
    identical inputs are preserved exactly); other tensors must be equal.
    Non-weight files (config, tokenizer, remote code) come from the last
    checkpoint. `output_dir` must not exist; it is written atomically, with
    `merge_manifest.json` recording the inputs.
    """
    paths = [Path(p).resolve() for p in checkpoints]
    output_dir = Path(output_dir).resolve()
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("Need at least two distinct checkpoints to average")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    config, specs = _config(paths[0]), _tensor_specs(paths[0])
    for path in paths[1:]:
        if _config(path) != config:
            raise ValueError(f"config.json differs from {paths[0]}: {path}")
        if _tensor_specs(path) != specs:
            raise ValueError(f"Tensor keys, shapes or dtypes differ: {path}")

    averaged = {}
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(p / WEIGHTS_NAME, framework="pt"))
            for p in paths
        ]
        for key in specs:
            tensors = [h.get_tensor(key) for h in handles]
            first = tensors[0]
            if first.is_floating_point():
                total = sum(t.to(torch.float64) for t in tensors)
                averaged[key] = (total / len(tensors)).to(first.dtype)
            elif all(torch.equal(first, t) for t in tensors[1:]):
                averaged[key] = first
            else:
                raise ValueError(f"Non-floating tensor {key} differs across inputs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}_", dir=output_dir.parent)
    )
    try:
        save_file(averaged, tmp_dir / WEIGHTS_NAME, metadata={"format": "pt"})
        for item in paths[-1].iterdir():
            if item.is_file() and not item.name.startswith(WEIGHTS_NAME):
                shutil.copy2(item, tmp_dir / item.name)
        manifest = {
            "method": "uniform",
            "checkpoints": [str(p) for p in paths],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (tmp_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
        tmp_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return output_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        type=Path,
        help="HF checkpoint directories to average (at least two).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to create for the averaged checkpoint; must not exist.",
    )
    args = parser.parse_args(argv)
    output_dir = average_checkpoints(args.checkpoints, args.output_dir)
    print(f"Averaged {len(args.checkpoints)} checkpoints into {output_dir}")


if __name__ == "__main__":
    main()
