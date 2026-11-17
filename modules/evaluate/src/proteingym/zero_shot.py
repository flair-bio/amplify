"""ProteinGym zero-shot scoring and reference-condition orchestration."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForMaskedLM

from modules.evaluate.src.artifacts.manifests import update_run_manifest
from modules.evaluate.src.artifacts.workspace import (
    EvaluationWorkspace,
    reserve_run_outputs,
)
from modules.evaluate.src.model.factory import (
    TORCH_DTYPES,
    build_masked_lm,
    load_tokenizer,
)
from modules.evaluate.src.proteingym.baselines import (
    score_assay_blosum62,
    score_assay_random,
)
from modules.evaluate.src.proteingym.config import ProteinGymZeroShotConfig
from modules.evaluate.src.proteingym.data import resolve_suite
from modules.evaluate.src.proteingym.masked_marginal import (
    score_assay_indel,
    score_assay_masked_marginal,
)
from modules.evaluate.src.proteingym.runner import run_assay_loop
from modules.evaluate.src.runners.runtime import Runtime

logger = logging.getLogger(__name__)


def run_zero_shot(
    config: ProteinGymZeroShotConfig, runtime: Runtime, *, overwrite: bool = False
) -> dict[str, object]:
    """Evaluate the benchmark using the application's shared runtime."""
    if config.workspace is None:
        return _run_zero_shot(config, runtime)
    workspace = EvaluationWorkspace(
        config.workspace,
        variant=config.run_variant(config.model_dump()),
        seed=config.seed,
    )
    settings = config.model_dump(
        mode="json", exclude={"model_id", "data_dir", "data_repo_id", "output_dir"}
    )
    with reserve_run_outputs(workspace, settings, overwrite=overwrite):
        resolved = config.model_copy(update={"data_dir": resolve_suite(config)})
        inputs = {"model_id": resolved.model_id, "data_dir": str(resolved.data_dir)}
        workspace.validate_experiment("inputs", inputs)
        update_run_manifest(workspace.run_manifest_path, "inputs", inputs)
        return _run_zero_shot(resolved, runtime)


def _score_assay_model(
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    batch_size: int,
    max_length: int | None,
    row: dict,
    assay: pl.DataFrame,
) -> np.ndarray:
    if row.get("is_indel"):
        return score_assay_indel(
            model,
            tokenizer,
            wt_sequence=row["target_seq"],
            mutated_sequences=assay["mutated_sequence"].to_list(),
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    return score_assay_masked_marginal(
        model,
        tokenizer,
        wt_sequence=row["target_seq"],
        mutants=assay["mutant"].to_list(),
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )


def _run_baseline(
    name: str,
    config: ProteinGymZeroShotConfig,
    tokenizer: object,
    device: torch.device,
    seed_model: torch.nn.Module | None = None,
) -> None:
    """Score every assay with a reference scorer, written under its own subfolder."""

    def score_assay(row: dict, assay: pl.DataFrame) -> np.ndarray:
        if name == "random":
            return score_assay_random(assay.height, seed=config.seed)
        if name == "blosum62":
            if row.get("is_indel"):
                return np.full(assay.height, np.nan)
            return score_assay_blosum62(assay["mutant"].to_list())
        return _score_assay_model(
            seed_model,
            tokenizer,
            device,
            config.batch_size,
            config.max_length,
            row,
            assay,
        )

    logger.info("Scoring baseline: %s", name)
    run_assay_loop(
        config.data_dir,
        config.output_dir / "baselines" / name,
        config.max_assays,
        score_assay,
        data_repo_id=config.data_repo_id,
        force_download=config.force_download,
        compute_extra_zero_shot_metrics=True,
        run_manifest_path=config.output_dir / "run_manifest.json"
        if config.workspace is not None
        else None,
        condition=name,
    )


def _run_zero_shot(
    config: ProteinGymZeroShotConfig, runtime: Runtime
) -> dict[str, object]:
    device = runtime.device
    tokenizer = load_tokenizer(config.model_id)
    if getattr(tokenizer, "remove_ambiguous", False):
        logger.info(
            "Disabling ambiguous-residue removal for ProteinGym scoring to preserve source residue coordinates."
        )
        tokenizer.remove_ambiguous = False
        tokenizer.init_kwargs["remove_ambiguous"] = False
    model = (
        build_masked_lm(
            config.model_id,
            **{
                "torch_dtype": TORCH_DTYPES[config.dtype],
                **(config.workspace.model_kwargs if config.workspace else {}),
            },
        )
        .to(device)
        .eval()
    )

    def score_assay(row: dict, assay: pl.DataFrame) -> np.ndarray:
        return _score_assay_model(
            model, tokenizer, device, config.batch_size, config.max_length, row, assay
        )

    summary = run_assay_loop(
        config.data_dir,
        config.output_dir,
        config.max_assays,
        score_assay,
        data_repo_id=config.data_repo_id,
        force_download=config.force_download,
        compute_uncertainty=config.compute_uncertainty,
        n_bootstrap=config.n_bootstrap,
        compute_extra_zero_shot_metrics=True,
        run_manifest_path=config.output_dir / "run_manifest.json"
        if config.workspace is not None
        else None,
    )
    for baseline in config.baselines:
        seed_model = None
        if baseline == "random_init_trunk":
            seed_model = (
                AutoModelForMaskedLM.from_config(model.config, trust_remote_code=True)
                .to(device)
                .eval()
            )
        _run_baseline(baseline, config, tokenizer, device, seed_model=seed_model)
    return summary
