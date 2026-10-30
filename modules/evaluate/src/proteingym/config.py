"""Config for the ProteinGym zero-shot and supervised scoring entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modules.evaluate.src.contracts.config import (
    EvaluationWorkspaceConfig,
    validate_directory_label,
)
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace

ProteinGymDtype = Literal["float32", "float16", "bfloat16"]


class ProteinGymWorkspaceConfig(EvaluationWorkspaceConfig):
    """Use the shared disk layout without registering an ordinary pipeline task."""

    task_type: str = "proteingym"
    target_type: Literal["continuous"] = "continuous"
    label_format: Literal["scalar"] = "scalar"
    dataset_name: str = "proteingym-dms_substitutions"
    model_name: str = "amplify-120m"
    run_id: str = "run01"

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        if value != "proteingym":
            raise ValueError("ProteinGym workspace.task_type must be 'proteingym'")
        return value

    @field_validator("dataset_config_name")
    @classmethod
    def _validate_dataset_subfolder(cls, value: str | None) -> str | None:
        return validate_directory_label(value) if value is not None else None

    @model_validator(mode="after")
    def _validate_sources(self) -> ProteinGymWorkspaceConfig:
        if not self.model_repo_id or not self.dataset_repo_id:
            raise ValueError(
                "ProteinGym workspace requires model_repo_id and dataset_repo_id"
            )
        source = Path(self.dataset_repo_id).expanduser()
        if self.dataset_revision is not None and (
            source.exists()
            or source.is_absolute()
            or self.dataset_repo_id.startswith((".", "~"))
        ):
            raise ValueError("workspace.dataset_revision must be null for local data")
        return self


class ProteinGymRunConfig(BaseModel):
    """Shared inputs; legacy flat configs retain their original direct paths."""

    model_config = ConfigDict(extra="forbid")

    workspace: ProteinGymWorkspaceConfig | None = None
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
    tf32: bool = True
    device: str | None = None
    seed: int = Field(1957723, ge=0)
    # Bootstrap SE of the aggregate score (see proteingym/aggregate.py); off by
    # default since it costs n_bootstrap resamples per run.
    compute_uncertainty: bool = False
    n_bootstrap: int = Field(10000, gt=0)

    @classmethod
    def run_variant(cls, values: dict[str, Any]) -> str:
        return "zero_shot"

    @model_validator(mode="before")
    @classmethod
    def _resolve_workspace_paths(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("workspace") is None:
            return value
        values = dict(value)
        workspace = ProteinGymWorkspaceConfig.model_validate(values["workspace"])
        seed = values.get("seed", cls.model_fields["seed"].default)
        run = get_evaluation_workspace(
            workspace, variant=cls.run_variant(values), seed=seed
        )
        subset = workspace.dataset_config_name or workspace.dataset_name
        resolved = {
            "model_id": workspace.model_repo_id,
            "data_dir": Path(workspace.dataset_repo_id or "").expanduser() / subset,
            "output_dir": run.run_root,
        }
        for key, path in resolved.items():
            if key in values and str(values[key]) != str(path):
                raise ValueError(
                    f"Do not mix workspace settings with legacy {key}; "
                    "use workspace overrides or set workspace=null for a flat config"
                )
            values[key] = path
        if values.get("data_repo_id") is not None:
            raise ValueError("Use workspace.dataset_repo_id instead of data_repo_id")
        values["workspace"] = workspace
        return values


class ProteinGymZeroShotConfig(ProteinGymRunConfig):
    """Config for ``run_proteingym_zero_shot.py``."""

    # Reference scorers run alongside the model, each written to its own
    # output_dir/baselines/<name>/ subfolder: "random" (chance-level sanity
    # check), "blosum62" (substitution-matrix baseline, substitutions only),
    # "random_init_trunk" (same architecture, randomly re-initialised weights).
    baselines: list[Literal["random", "blosum62", "random_init_trunk"]] = []


class ProteinGymSupervisedConfig(ProteinGymRunConfig):
    """Config for ``run_proteingym_supervised.py``."""

    # Official fold columns are sourced with the assay; "seeded" uses the
    # persisted train/validation/test split generated during sourcing.
    # "official_average" averages all three official DMS schemes for
    # substitutions and uses only fold_random_5 for indels, matching
    # performance_DMS_supervised_benchmarks.py.
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

    @classmethod
    def run_variant(cls, values: dict[str, Any]) -> str:
        probe = values.get("probe_type", cls.model_fields["probe_type"].default)
        split = values.get("split", cls.model_fields["split"].default)
        return f"{probe}-frozen_trunk-{split}"
