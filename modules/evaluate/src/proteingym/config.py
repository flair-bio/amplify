"""Config for the ProteinGym zero-shot and supervised scoring entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProteinGymDtype = Literal["float32", "float16", "bfloat16"]


class ProteinGymZeroShotConfig(BaseModel):
    """Config for ``run_proteingym_zero_shot.py``."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    # Final fallback directory containing reference.parquet plus
    # data/<dms_id>.parquet per assay, and the download target when HF data is
    # unavailable locally.
    data_dir: Path
    # HF dataset repo to prefer before falling back to data_dir. A local HF
    # snapshot is tried first, then the repo is downloaded if needed.
    data_repo_id: str | None = None
    # Skip the local HF snapshot and force a fresh download; data_dir remains
    # the fallback if that download is unavailable.
    force_download: bool = False
    output_dir: Path
    batch_size: int = Field(16, gt=0)
    max_length: int | None = 1024
    max_assays: int | None = Field(default=None, gt=0)
    dtype: ProteinGymDtype = "float32"
    device: str | None = None
    seed: int = 1957723
    # Bootstrap SE of the aggregate score (see proteingym/aggregate.py); off by
    # default since it costs n_bootstrap resamples per run.
    compute_uncertainty: bool = False
    n_bootstrap: int = Field(10000, gt=0)
    # Reference scorers run alongside the model, each written to its own
    # output_dir/baselines/<name>/ subfolder: "random" (chance-level sanity
    # check), "blosum62" (substitution-matrix baseline, substitutions only),
    # "random_init_trunk" (same architecture, randomly re-initialised weights).
    baselines: list[Literal["random", "blosum62", "random_init_trunk"]] = []


class ProteinGymSupervisedConfig(BaseModel):
    """Config for ``run_proteingym_supervised.py``."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    # Final fallback directory containing reference.parquet plus
    # data/<dms_id>.parquet per assay, and the download target when HF data is
    # unavailable locally.
    data_dir: Path
    # HF dataset repo to prefer before falling back to data_dir. A local HF
    # snapshot is tried first, then the repo is downloaded if needed.
    data_repo_id: str | None = None
    # Skip the local HF snapshot and force a fresh download; data_dir remains
    # the fallback if that download is unavailable.
    force_download: bool = False
    output_dir: Path
    # Official fold columns are sourced with the assay; "seeded" uses the
    # persisted train/validation/test split generated during sourcing.
    # "official_average" runs all three official fold columns per assay and
    # averages the resulting metric equally, matching the official supervised
    # leaderboard protocol (performance_DMS_supervised_benchmarks.py) instead
    # of reporting a single fold scheme.
    split: Literal[
        "fold_random_5",
        "fold_modulo_5",
        "fold_contiguous_5",
        "official_average",
        "seeded",
        "custom_kfold",
    ] = "fold_random_5"
    # Number of folds generated when split='custom_kfold'.
    custom_kfold_splits: int = Field(5, gt=1)
    # "linear" fits Ridge/LogisticRegression; "mlp" fits a two-layer
    # Linear-GELU-Linear torch MLP by gradient descent (matching
    # EmbeddingProbeHead's projection+classifier shape); "torch_linear" fits a
    # single torch.nn.Linear layer by gradient descent.
    probe_type: Literal["linear", "mlp", "torch_linear"] = "linear"
    # Fixed constructor kwargs for the chosen probe (merged under any
    # search_space candidate). Ignored keys raise from the estimator itself.
    probe_hyperparameters: dict = {}
    # Hyperparameter grid swept via sklearn's ParameterGrid, scored on an
    # inner holdout carved out of each fold's training rows. Empty means no
    # sweep: fit once with probe_hyperparameters.
    probe_search_space: dict = {}
    batch_size: int = Field(8, gt=0)
    max_length: int | None = 1024
    max_assays: int | None = Field(default=None, gt=0)
    dtype: ProteinGymDtype = "float32"
    device: str | None = None
    seed: int = 1957723
    compute_uncertainty: bool = False
    n_bootstrap: int = Field(10000, gt=0)
