"""One-pass MLM plus latent-alignment lifecycle, diagnostics and native export."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from modules.post_train.src.config import PostTrainingConfig
from modules.post_train.src.dataset.collator import collate_records
from modules.post_train.src.dataset.manifest import (
    load_dataset,
)
from modules.post_train.src.model.alignment import (
    AlignmentModel,
    export_student,
    load_student,
)
from modules.post_train.src.model.tokenization import load_vocabulary
from modules.post_train.src.trainers.latent_alignment import (
    cached_backward,
    gradient_norms,
    make_optimizer,
    seed_everything,
    validation_mlm,
)
from modules.post_train.src.utils.tracking import (
    log_wandb,
    wandb_metrics,
    wandb_run,
)

logger = logging.getLogger(__name__)


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
    device = resolve_device(options.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("CUDA training requires native BF16 support.")
    logger.info("Using device: %s", device)
    logger.info("Loading dataset and checking embedding files: %s", config.dataset.path)
    prepared = load_dataset(config.dataset)
    train, validation = prepared.train, prepared.validation
    logger.info(
        "Dataset ready: %d training proteins, %d validation proteins, %d embedding features",
        len(train),
        len(validation),
        prepared.embedding_spec.dimension,
    )
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
    logger.info("Loading student checkpoint: %s", checkpoint)
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
    provenance = {
        "manifest": str(prepared.manifest),
        "hub_revision": prepared.hub_revision,
        "dataset_metadata": prepared.metadata,
        "student_checkpoint": str(checkpoint),
    }
    run_record: dict[str, Any] = {
        "status": "running",
        "config": resolved,
        "provenance": provenance,
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run_record, indent=2, allow_nan=False) + "\n")
    order = torch.randperm(
        len(train), generator=torch.Generator().manual_seed(config.seed)
    ).tolist()
    steps = len(train) // options.batch_size
    logger.info(
        "Starting %d updates (batch %d, pool %d, microbatch %d); output: %s",
        steps,
        options.batch_size,
        alignment.pool_proteins,
        options.microbatch_size,
        output,
    )
    final_training_metrics: dict[str, Any] = {}
    final_validation_metrics: dict[str, Any] = {}
    started = time.perf_counter()
    model.train()
    with (
        wandb_run(config, output, resolved, device),
        tqdm(
            total=steps,
            desc="Starting",
            unit="update",
            dynamic_ncols=True,
        ) as progress,
    ):
        latest_validation_mlm: float | None = None
        progress.set_description_str("Validating (initial)")
        initial_validation = validation_mlm(
            model,
            validation,
            config.seed,
            device,
            vocabulary=vocabulary,
            mlm=config.mlm,
        )
        initial_row = {"step": 0, "validation": initial_validation}
        log_wandb(wandb_metrics(initial_row), step=initial_row["step"], device=device)
        latest_validation_mlm = initial_validation["mlm"]
        progress.set_description_str("Post-training")
        progress.set_postfix({"val_mlm": f"{latest_validation_mlm:.4f}"}, refresh=False)
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
                    mlm_weight=config.mlm.loss_weight,
                    anchor_chunk_size=options.anchor_chunk_size,
                    verify_replay=step == 0,
                )
            )
            if step == 0 and config.wandb.enabled:
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
                progress.set_description_str(f"Validating (step {step + 1})")
                row["validation"] = validation_mlm(
                    model,
                    validation,
                    config.seed,
                    device,
                    vocabulary=vocabulary,
                    mlm=config.mlm,
                )
            final_training_metrics = {
                key: row[key] for key in ("mlm", "latent", "total")
            }
            if "validation" in row:
                final_validation_metrics = row["validation"]
                latest_validation_mlm = row["validation"]["mlm"]
            log_wandb(wandb_metrics(row), step=row["step"], device=device)
            postfix = {
                key: f"{row[key]:.4f}"
                for key in ("total", "mlm", "latent")
                if key in row
            }
            if latest_validation_mlm is not None:
                postfix["val_mlm"] = f"{latest_validation_mlm:.4f}"
            progress.set_description_str("Post-training")
            progress.set_postfix(postfix, refresh=False)
            progress.update(row["step"] - progress.n)
            del batches
        progress.set_description_str("Exporting")
        export = export_student(model, checkpoint, output)
        run_record["checkpoint"] = {
            "path": str(export),
            "selection": "final_after_one_epoch",
        }
        run_record["results"] = {
            "training": {
                "steps": steps,
                "proteins": len(train),
                "residues": sum(len(row.sequence) for row in train),
                "metrics": final_training_metrics,
            },
            "validation": final_validation_metrics,
        }
        run_record["runtime"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        }
        run_record["status"] = "completed"
        run_path.write_text(json.dumps(run_record, indent=2, allow_nan=False) + "\n")
        progress.set_description_str("Completed")
    logger.info("Exported checkpoint: %s", export)
    return run_record
