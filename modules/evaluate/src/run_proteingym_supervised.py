"""Entry point for ProteinGym supervised scoring: pool frozen-trunk embeddings
per variant, fit a linear probe per CV fold (official or custom), and report
out-of-fold Spearman (DMS) / AUC (clinical) per assay plus the ProteinGym
aggregation.

Deliberately separate from run_evaluate.py's tune_probe pipeline: ProteinGym's
protocol is fixed per-assay CV folds with out-of-fold scoring, not a single
train/validation/test split.

Usage::

    uv run modules/evaluate/src/run_proteingym_supervised.py <config.yaml> [key=value ...]
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from modules.evaluate.src.dataset_sourcing.pipeline import (
    load_runtime_config_from_argv,
)
from modules.evaluate.src.proteingym.aggregate import compute_assay_metric
from modules.evaluate.src.proteingym.config import ProteinGymSupervisedConfig
from modules.evaluate.src.proteingym.runner import run_assay_loop
from modules.evaluate.src.proteingym.supervised import (
    OFFICIAL_CV_SCHEMES,
    pool_embeddings,
    resolve_folds,
    run_cv,
    run_seeded_split,
)
from modules.evaluate.src.proteingym.supervised import Task
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


def run(config: ProteinGymSupervisedConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            torch_dtype=_TORCH_DTYPE[config.dtype],
        )
        .to(device)
        .eval()
    )

    def score_assay(
        row: dict, assay: pl.DataFrame
    ) -> np.ndarray | tuple[np.ndarray, float]:
        has_score = "dms_score" in assay.columns
        task: Task = "regression" if has_score else "classification"
        targets = (
            assay["dms_score"].to_numpy()
            if has_score
            else assay["dms_score_bin"].to_numpy()
        )

        embeddings = pool_embeddings(
            model,
            tokenizer,
            assay["mutated_sequence"].to_list(),
            device=device,
            batch_size=config.batch_size,
            max_length=config.max_length,
        )
        if config.split == "seeded":
            if "split" not in assay.columns:
                raise ValueError(
                    "split='seeded' requires sourced train/validation/test labels"
                )
            return run_seeded_split(
                embeddings,
                targets,
                assay["split"].to_numpy(),
                task=task,
                probe_type=config.probe_type,
                hyperparameters=config.probe_hyperparameters,
                search_space=config.probe_search_space,
                seed=config.seed,
                device=device,
            )
        if config.split == "official_average":
            # The official supervised benchmark only averages fold_random_5 for
            # indels (fold_modulo_5/fold_contiguous_5 aren't computed for them);
            # substitutions average all three official CV schemes equally.
            schemes = ("fold_random_5",) if row.get("is_indel") else OFFICIAL_CV_SCHEMES
            missing = [scheme for scheme in schemes if scheme not in assay.columns]
            if missing:
                raise ValueError(
                    "split='official_average' requires official fold columns "
                    f"{missing}; re-source the assay with the substitutions track"
                )
            metric_values = []
            predictions_by_scheme = {}
            for scheme in schemes:
                predictions = run_cv(
                    embeddings,
                    targets,
                    assay[scheme].to_numpy(),
                    task=task,
                    probe_type=config.probe_type,
                    hyperparameters=config.probe_hyperparameters,
                    search_space=config.probe_search_space,
                    seed=config.seed,
                    device=device,
                )
                _, value = compute_assay_metric(
                    predictions,
                    dms_scores=targets if has_score else None,
                    dms_score_bin=targets if not has_score else None,
                    flip_binary_labels=bool(row.get("is_clinical")),
                )
                metric_values.append(value)
                predictions_by_scheme[scheme] = predictions
            # The scored parquet stores the per-variant mean prediction across
            # schemes (not just one scheme's), so it stays representative of
            # the reported metric, which is the equal average across schemes.
            mean_predictions = np.nanmean(
                np.stack(list(predictions_by_scheme.values())), axis=0
            )
            return mean_predictions, float(np.nanmean(metric_values))
        official_folds = (
            assay[config.split].to_numpy() if config.split in assay.columns else None
        )
        folds = resolve_folds(
            assay.height,
            official_folds,
            config.split,
            n_splits=config.custom_kfold_splits,
            seed=config.seed,
        )
        return run_cv(
            embeddings,
            targets,
            folds,
            task=task,
            probe_type=config.probe_type,
            hyperparameters=config.probe_hyperparameters,
            search_space=config.probe_search_space,
            seed=config.seed,
            device=device,
        )

    return run_assay_loop(
        config.data_dir,
        config.output_dir,
        config.max_assays,
        score_assay,
        data_repo_id=config.data_repo_id,
        force_download=config.force_download,
        compute_uncertainty=config.compute_uncertainty,
        n_bootstrap=config.n_bootstrap,
    )


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: run_proteingym_supervised.py <config.yaml> [key=value ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    config = load_runtime_config_from_argv(args, model_cls=ProteinGymSupervisedConfig)
    run(config)


if __name__ == "__main__":
    main()
