"""One-pass MLM plus latent-alignment lifecycle, diagnostics and native export."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modules.post_train.src.contracts.config import PostTrainingConfig
from modules.post_train.src.dataset.collator import collate_records
from modules.post_train.src.dataset.latent_alignment import (
    load_dataset,
)
from modules.post_train.src.model.alignment import (
    AlignmentModel,
    export_student,
    load_student,
    native_assets,
)
from modules.post_train.src.model.tokenization import load_vocabulary
from modules.post_train.src.tracking.progress import track_run
from modules.post_train.src.trainers.latent_alignment import (
    cached_backward,
    gradient_norms,
    make_optimizer,
    seed_everything,
    validation_mlm,
)


def checkpoint_checksums(checkpoint: Path) -> dict[str, str]:
    """Pin initial tensors as well as configuration and executable assets."""
    paths = {checkpoint / name for name in (*native_assets(checkpoint), "config.json")}
    for pattern in ("*.safetensors", "*.bin", "*.index.json"):
        paths.update(checkpoint.glob(pattern))
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        result[path.name] = digest.hexdigest()
    return result


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if not torch.cuda.is_available():
            raise ValueError(
                "No CUDA GPU is available. Run on a GPU allocation or explicitly "
                "set training.device=cpu for bounded checks."
            )
        return torch.device("cuda:0")
    return torch.device(requested)


def run(config: PostTrainingConfig) -> dict[str, Any]:
    options = config.training
    alignment = config.alignment
    output = config.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(options.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("CUDA training requires native BF16 support.")
    torch.use_deterministic_algorithms(True)
    prepared = load_dataset(config.dataset)
    train, validation = prepared.train, prepared.validation
    if len(train) % options.batch_size:
        raise ValueError(
            "Training count must divide batch_size exactly: no dropping or repetition."
        )
    checkpoint = config.student.checkpoint.expanduser().resolve(strict=True)
    for input_root in (checkpoint, prepared.manifest.parent):
        if (
            output == input_root
            or output in input_root.parents
            or input_root in output.parents
        ):
            raise ValueError(
                "Output and input data/checkpoint directories must not be nested."
            )
    seed_everything(config.seed)
    vocabulary = load_vocabulary(
        checkpoint,
        residues={aa for row in train + validation for aa in row.sequence},
    )
    student = load_student(
        checkpoint,
        device,
        trust_remote_code=config.student.trust_remote_code,
        required_positions=max(len(row.sequence) for row in train + validation) + 2,
        vocabulary=vocabulary,
    )
    model = AlignmentModel(
        student,
        alignment,
        teacher_dim=prepared.embedding_spec.dimension,
    ).to(device)
    optimizer = make_optimizer(model, options)
    seed_everything(config.seed)
    output.mkdir(parents=True, exist_ok=False)
    resolved = config.model_dump(mode="json")
    resolved["training"]["device"] = str(device)
    resolved["output_dir"] = str(output)
    resolved["student"]["checkpoint"] = str(checkpoint)
    if prepared.hub_revision is None:
        resolved["dataset"]["path"] = str(
            Path(config.dataset.path).expanduser().absolute()
        )
    (output / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    provenance = {
        "manifest": str(prepared.manifest),
        "manifest_sha256": prepared.manifest_sha256,
        "hub_revision": prepared.hub_revision,
        "dataset_metadata": prepared.metadata,
        "student_config_sha256": hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "student_checkpoint": str(checkpoint),
        "student_files_sha256": checkpoint_checksums(checkpoint),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    steps = len(train) // options.batch_size
    started = time.perf_counter()
    model.train()
    with track_run(config, output, resolved, steps, device) as logger:
        logger.phase("Validating (initial)")
        logger.log(
            {
                "step": 0,
                "validation": validation_mlm(
                    model,
                    validation,
                    config.seed,
                    device,
                    vocabulary=vocabulary,
                    mlm=config.mlm,
                ),
            }
        )
        for step in range(steps):
            begin = time.perf_counter()
            indices = order[step * options.batch_size : (step + 1) * options.batch_size]
            batches = [
                collate_records(
                    [
                        train[index]
                        for index in indices[micro : micro + options.microbatch_size]
                    ],
                    torch.Generator().manual_seed(
                        config.seed + step * 1000003 + micro // options.microbatch_size
                    ),
                    device,
                    vocabulary=vocabulary,
                    mlm=config.mlm,
                )
                for micro in range(0, options.batch_size, options.microbatch_size)
            ]
            optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, Any] = dict(
                cached_backward(
                    model,
                    batches,
                    alignment.pool_proteins // options.microbatch_size,
                    anchor_chunk_size=options.anchor_chunk_size,
                    verify_replay=step == 0,
                )
            )
            if step == 0:
                metrics["gradient_norms_before_clipping"] = gradient_norms(model)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), options.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "step": step + 1,
                **metrics,
                "gradient_norm": float(norm),
                "step_seconds": time.perf_counter() - begin,
                "encoder_lr": options.encoder_lr,
                "head_lr": options.head_lr,
            }
            if (step + 1) % options.validation_interval == 0 or step + 1 == steps:
                logger.phase(f"Validating (step {step + 1})")
                row["validation"] = validation_mlm(
                    model,
                    validation,
                    config.seed,
                    device,
                    vocabulary=vocabulary,
                    mlm=config.mlm,
                )
            logger.log(row)
            del batches
        logger.phase("Exporting")
        export = export_student(model, checkpoint, output)
        import transformers

        summary = {
            "completed": True,
            "steps": steps,
            "epochs": 1,
            "train_proteins": len(train),
            "validation_proteins": len(validation),
            "protein_presentations": len(train),
            "residue_presentations": sum(len(row.sequence) for row in train),
            "batch_size": options.batch_size,
            "negative_pool_proteins": alignment.pool_proteins,
            "independent_pools_per_update": options.batch_size
            // alignment.pool_proteins,
            "student_parameters": sum(
                parameter.numel() for parameter in student.parameters()
            ),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "transformers_version": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "elapsed_seconds": time.perf_counter() - started,
            "export": str(export),
            "provenance": provenance,
            "method": {
                "mlm_weight": 1.0,
                "latent_weight": alignment.loss_weight,
                "projection_dimension": alignment.projection_dim,
                "embedding_dimension": prepared.embedding_spec.dimension,
                "vocabulary_size": vocabulary.size,
                "sequence_features": "last_block_residual_before_final_norm",
                "mask_probability": config.mlm.mask_probability,
                "mask_replacement": {
                    "mask": config.mlm.mask_replace_probability,
                    "uniform_vocabulary": config.mlm.random_replace_probability,
                    "unchanged": 1
                    - config.mlm.mask_replace_probability
                    - config.mlm.random_replace_probability,
                },
                "contrastive_loss": "symmetric_residue_InfoNCE_all_offdiagonal_negatives",
                "initial_inverse_temperature": 1 / alignment.initial_temperature,
                "inverse_temperature_lower_bound": alignment.min_inverse_temperature,
                "schedule": "constant_no_warmup",
                "optimizer": "AdamW",
                "betas": list(options.betas),
                "weight_decay": options.weight_decay,
                "gradient_clip": options.gradient_clip,
            },
            "checkpoint_selection": "fixed_one_epoch_final",
            "resume_supported": False,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        logger.phase("Completed")
        return summary
