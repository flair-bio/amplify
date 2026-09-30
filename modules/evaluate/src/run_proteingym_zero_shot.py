"""Entry point for ProteinGym zero-shot scoring: DMS or clinical, substitutions
or indels (dispatched per assay from the sourced ``is_indel`` reference flag).

Deliberately separate from run_evaluate.py's Prepare/Tune/Predict/Score
pipeline: there's no training, no held-out split, and the official ProteinGym
metric aggregation
(mean over UniProt IDs, then over function categories) doesn't fit
ScoreStep's per-example bootstrap contract.

Usage::

    uv run modules/evaluate/src/run_proteingym_zero_shot.py <config.yaml> [key=value ...]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from modules.evaluate.src.dataset_sourcing.pipeline import (
    load_runtime_config_from_argv,
)
from modules.evaluate.src.proteingym.baselines import (
    score_assay_blosum62,
    score_assay_random,
)
from modules.evaluate.src.proteingym.config import ProteinGymZeroShotConfig
from modules.evaluate.src.proteingym.masked_marginal import (
    score_assay_indel,
    score_assay_masked_marginal,
)
from modules.evaluate.src.proteingym.runner import run_assay_loop
from modules.evaluate.src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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
        # name == "random_init_trunk"
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
    )


def run(config: ProteinGymZeroShotConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    model = (
        AutoModelForMaskedLM.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=_TORCH_DTYPE[config.dtype],
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
    )

    for baseline in config.baselines:
        seed_model = None
        if baseline == "random_init_trunk":
            # Same architecture, freshly initialised (untrained) weights.
            seed_model = (
                AutoModelForMaskedLM.from_config(model.config, trust_remote_code=True)
                .to(device)
                .eval()
            )
        _run_baseline(baseline, config, tokenizer, device, seed_model=seed_model)

    return summary


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: run_proteingym_zero_shot.py <config.yaml> [key=value ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_runtime_config_from_argv(args, model_cls=ProteinGymZeroShotConfig)
    run(config)


if __name__ == "__main__":
    main()
