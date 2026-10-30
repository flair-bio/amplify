"""On-disk layout for one (model, dataset) evaluation workspace and its run variants."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modules.evaluate.src.artifacts.identity import (
    dataset_identity,
    model_identity,
    validate_dataset_pin,
    validate_model_pin,
)
from modules.evaluate.src.tasks import TaskType, available_tasks
from modules.evaluate.src.utils.io import update_run_manifest, write_json

__all__ = [
    "DatasetWorkspaceConfig",
    "EvaluationWorkspace",
    "EvaluationWorkspaceConfig",
    "TaskType",
    "get_evaluation_workspace",
]


class DatasetWorkspaceConfig(BaseModel):
    """Dataset/task settings shared by model and baseline runs."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str  # dataset name that serves as a mnemonic for the dataset being evaluated, e.g. "stability_prediction"
    dataset_repo_id: str | None = (
        None  # path to HuggingFace dataset repo e.g. biomap-research/stability_prediction
    )
    dataset_revision: str | None = None
    # Named view within a shared repo (HF builder config / local subdirectory).
    dataset_subset: str | None = None
    # Relative directory under --data-root (defaults to dataset_name there).
    local_path: str | None = None
    edition: Literal[14, 15, 16] | None = None
    task_type: TaskType
    target_type: Literal["categorical", "continuous"]
    label_format: Literal[
        "scalar", "per_token", "multilabel", "contact_pairs", "consensus_profile"
    ]
    num_labels: int | None = (
        None  # number of classes for *_classification tasks; unused (forced to 1) for sequence_regression
    )
    base_path: str = "./eval"  # base path where the EvaluationWorkspace is stored
    output_dir: Path | None = None

    @field_validator("dataset_name")
    @classmethod
    def _validate_dataset_label(cls, value: str) -> str:
        return validate_directory_label(value)

    @property
    def benchmark_root(self) -> Path:
        return self.output_dir if self.output_dir is not None else Path(self.base_path)

    @property
    def dataset_root(self) -> Path:
        return self.benchmark_root / self.task_type / self.dataset_name

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, value: str) -> str:
        if value not in available_tasks():
            raise ValueError(
                f"Unknown task_type '{value}'. Available: {available_tasks()}."
            )
        return value

    @model_validator(mode="after")
    def _validate_dataset_pin(self) -> DatasetWorkspaceConfig:
        # Offline (no Hub or data access): same rules as the loaders enforce later.
        validate_dataset_pin(
            task_type=self.task_type,
            repo_id=self.dataset_repo_id,
            revision=self.dataset_revision,
            subset=self.dataset_subset,
        )
        return self

    @model_validator(mode="after")
    def _validate_target_metadata(self) -> DatasetWorkspaceConfig:
        if self.target_type == "continuous":
            if self.label_format != "scalar" or self.num_labels is not None:
                raise ValueError(
                    "target_type='continuous' requires label_format='scalar' and "
                    "num_labels=null."
                )
        elif self.label_format in {"scalar", "per_token", "multilabel"} and (
            self.num_labels is None or self.num_labels <= 1
        ):
            raise ValueError("Categorical targets require num_labels > 1.")
        return self


class EvaluationWorkspaceConfig(DatasetWorkspaceConfig):
    """Paths and model settings for one model/dataset evaluation."""

    model_name: str
    run_id: str
    model_repo_id: str | None = None
    # Immutable commit SHA for Hub models; fingerprinted local paths take none.
    model_revision: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tmp_path: str | None = None

    @model_validator(mode="after")
    def _validate_model_pin(self) -> EvaluationWorkspaceConfig:
        validate_model_pin(repo_id=self.model_repo_id, revision=self.model_revision)
        return self

    @field_validator("run_id", "model_name")
    @classmethod
    def _validate_directory_label(cls, value: str) -> str:
        return validate_directory_label(value)


def validate_directory_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "Output labels must start with a letter or digit and contain "
            "only letters, digits, '.', '_', or '-'."
        )
    return value


class EvaluationWorkspace:
    """Resolve paths for one (model, dataset) pair.

    Embeddings are shared across runs. Each run owns its predictions, scores,
    fitted controls, and manifest.
    """

    def __init__(
        self,
        config: EvaluationWorkspaceConfig,
        variant: str | None = None,
        seed: int | None = None,
    ) -> None:

        self.config = config
        self.variant = variant
        self.seed = seed

        self.dataset_name = config.dataset_name
        self.dataset_repo_id = config.dataset_repo_id
        self.model_name = config.model_name
        self.model_repo_id = config.model_repo_id
        self.task_type = config.task_type
        self.num_labels = config.num_labels
        self.model_kwargs = config.model_kwargs
        # Pinned Hub revision, forwarded to every tokenizer/model load so the
        # trunk, tokenizer and random-trunk controls all come from one commit.
        self.model_load_kwargs: dict[str, Any] = (
            {"revision": config.model_revision}
            if config.model_revision is not None
            else {}
        )
        self._pinned_identity: dict[str, Any] | None = None
        self.base_path = config.base_path
        if variant is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", variant
        ):
            raise ValueError("Run variants must be safe single-directory labels.")
        workspace_root = config.dataset_root / "models" / config.model_name
        self.workspace_root = workspace_root
        self.tmp_path = (
            Path(config.tmp_path) if config.tmp_path else workspace_root / "tmp"
        )
        # Disposable: safe to delete between runs, PrepareStep recomputes on miss.
        # Deliberately NOT nested under `variant`: a frozen trunk's cached
        # embeddings don't depend on head_type/mode, so every variant
        # sweep for this model+dataset can reuse the same cache.
        self.embeddings_path = self.tmp_path / "embeddings"

        self.run_name = f"{variant}-{config.run_id}" if variant else config.run_id
        run_root = workspace_root / self.run_name
        self.experiment_root = run_root
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Run seed must be a nonnegative integer")
            run_root = run_root / f"seed_{seed}"
        self.model_path = run_root / "model"
        self.run_root = run_root
        self.preds_path = run_root
        self.stats_path = run_root
        self.scores_path = run_root

        # Consolidated Predict/Score settings + results sidecar (replaces
        # separate predict_config.json/scores.json/bootstrap_summary.json
        # files). Model checkpoint settings are deliberately NOT included
        # here: they stay beside the checkpoint in `model_path` (see
        # TuneStep._save_model) so it remains self-describing if copied out
        # on its own.
        self.run_manifest_path = run_root / "run_manifest.json"

        self.comparison_path = run_root

        self._all_paths = [run_root]

    def control_workspace(self, name: str) -> EvaluationWorkspace:
        """Keep a fitted control and its predictions inside the owning run."""
        validate_directory_label(name)
        workspace = EvaluationWorkspace(self.config, self.variant, self.seed)
        workspace.run_root = self.run_root / "controls" / name
        workspace.model_path = workspace.run_root / "model"
        workspace.preds_path = workspace.run_root
        workspace.stats_path = workspace.run_root
        workspace.scores_path = workspace.run_root
        workspace.comparison_path = workspace.run_root
        workspace.run_manifest_path = workspace.run_root / "run_manifest.json"
        workspace._all_paths = [workspace.run_root]
        return workspace

    def setup_directories(self) -> None:
        """Create all required directories on disk."""
        for path in self._all_paths:
            path.mkdir(parents=True, exist_ok=True)

    def pinned_identity(self) -> dict[str, Any]:
        """Model checkpoint and dataset identity, hashed once per workspace."""
        if self._pinned_identity is None:
            self._pinned_identity = {
                "model": model_identity(
                    label=self.model_name,
                    source=self.model_repo_id or self.model_name,
                    requested_revision=self.config.model_revision,
                ),
                "dataset": dataset_identity(
                    source=self.dataset_repo_id or self.dataset_name,
                    subset=self.config.dataset_subset,
                    revision=self.config.dataset_revision,
                ),
            }
        return self._pinned_identity

    @contextmanager
    def reserve_run(self, *, overwrite: bool = False) -> Iterator[list[Path]]:
        """Hold a nonblocking lock until this ordinary pipeline finishes."""
        if self.variant is None:
            raise ValueError("Pipeline output protection requires a run variant.")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        suffix = "" if self.seed is None else f"-seed_{self.seed}"
        lock_path = self.workspace_root / f".{self.run_name}{suffix}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError(
                    f"Evaluation is already running for {self.run_root}."
                ) from error
            provenance_path = self.workspace_root / "workspace_manifest.json"
            provenance = {
                "dataset_name": self.dataset_name,
                "dataset_repo_id": self.dataset_repo_id,
                "task_type": self.task_type,
                "model_name": self.model_name,
                "model_repo_id": self.model_repo_id,
            }
            identity = self.pinned_identity()
            with (self.workspace_root / ".workspace.lock").open("a") as workspace_lock:
                fcntl.flock(workspace_lock, fcntl.LOCK_EX)
                if provenance_path.exists():
                    recorded = json.loads(provenance_path.read_text())
                    if {key: recorded.get(key) for key in provenance} != provenance:
                        raise ValueError(
                            f"Output labels already identify different inputs at "
                            f"{self.workspace_root}; choose a new model/dataset label or benchmark root."
                        )
                    # Same labels but a different checkpoint, revision or subset
                    # would silently merge two experiments into one directory.
                    if "identity" in recorded and recorded["identity"] != identity:
                        raise ValueError(
                            f"Output labels at {self.workspace_root} were recorded for a "
                            "different model checkpoint, dataset revision or subset; "
                            "choose a new model or dataset label."
                        )
                    if "identity" not in recorded:
                        # Adopting an identity-less manifest would bind these
                        # outputs to whatever checkpoint is present now, which
                        # cannot be verified. Refuse and leave outputs untouched.
                        raise ValueError(
                            f"{self.workspace_root} holds outputs from a run that "
                            "predates checkpoint identity recording, so they cannot be "
                            "verified against the current model or dataset. Nothing was "
                            "changed. To proceed, move this benchmark directory aside "
                            "(or choose a new model/dataset label or run_id) and rerun; "
                            "existing results are never overwritten or re-adopted."
                        )
                else:
                    write_json(provenance_path, {**provenance, "identity": identity})
            runs_path = self.workspace_root
            if runs_path.is_symlink():
                raise ValueError(
                    f"Refusing to modify symlinked run outputs: {runs_path}"
                )
            dependent_pattern = re.compile(
                rf"{re.escape(self.variant)}-layer_\d+-{re.escape(self.config.run_id)}"
            )
            targets = [self.run_root]
            if (
                self.seed is None
                and (self.experiment_root / "experiment_config.json").exists()
            ):
                raise ValueError(
                    "Seeded outputs already use this run_id; select a seed or a new run_id"
                )
            if (
                self.seed is not None
                and (self.experiment_root / "run_manifest.json").exists()
            ):
                raise ValueError(
                    "Unseeded outputs already use this run_id; choose a new run_id"
                )
            if runs_path.exists():
                targets.extend(
                    path if self.seed is None else path / f"seed_{self.seed}"
                    for path in runs_path.iterdir()
                    if dependent_pattern.fullmatch(path.name)
                )
            for target in targets:
                if target.is_symlink() or target.parent.is_symlink():
                    raise ValueError(f"Refusing to modify symlinked outputs: {target}")
                for source in (
                    self.tmp_path,
                    Path(self.model_repo_id) if self.model_repo_id else None,
                    Path(self.dataset_repo_id) if self.dataset_repo_id else None,
                ):
                    if source is not None and source.resolve().is_relative_to(
                        target.resolve()
                    ):
                        raise ValueError(
                            f"Output target {target} overlaps input/cache path {source}."
                        )
                if target.exists() and any(target.iterdir()) and not overwrite:
                    raise FileExistsError(
                        f"Evaluation outputs already exist at {target}. "
                        "Choose another workspace.run_id or use --overwrite "
                        "to replace this variant and its derived outputs."
                    )
            yield targets

    def initialize_run(
        self, targets: list[Path], *, overwrite: bool, config: dict[str, Any]
    ) -> None:
        """Clean only reserved variant outputs, preserving caches and siblings."""
        if self.seed is not None:
            identity = {key: value for key, value in config.items() if key != "seed"}
            self.validate_experiment("config", identity)
        if overwrite:
            runs_path = self.workspace_root
            pattern = re.compile(
                rf"{re.escape(self.variant or '')}(?:-layer_\d+)?-{re.escape(self.config.run_id)}"
            )
            for target in targets:
                if (
                    (
                        target.parent != runs_path
                        if self.seed is None
                        else (
                            target.parent.parent != runs_path
                            or target.name != f"seed_{self.seed}"
                        )
                    )
                    or not pattern.fullmatch(
                        target.name if self.seed is None else target.parent.name
                    )
                    or target.is_symlink()
                ):
                    raise ValueError(
                        f"Refusing to remove unreserved output path: {target}"
                    )
            for target in targets:
                if target.exists():
                    shutil.rmtree(target)
        update_run_manifest(
            self.run_manifest_path,
            "evaluation",
            {"status": "running", "config": config},
        )

    def validate_experiment(self, kind: str, identity: dict[str, Any]) -> None:
        """Bind sibling seed runs to identical settings and immutable inputs."""
        if self.experiment_root.is_symlink() or self.workspace_root.is_symlink():
            raise ValueError("Refusing to modify symlinked experiment outputs")
        validate_directory_label(kind)
        path = self.experiment_root / f"experiment_{kind}.json"
        lock_path = self.workspace_root / f".{self.run_name}-experiment.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_symlink():
                raise ValueError("Refusing to modify symlinked experiment identity")
            if path.exists():
                if json.loads(path.read_text()) != identity:
                    raise ValueError(
                        f"Experiment {kind} differs across seeds; choose a new run_id"
                    )
            else:
                self.experiment_root.mkdir(parents=True, exist_ok=True)
                write_json(path, identity)


def get_evaluation_workspace(
    config: EvaluationWorkspaceConfig,
    variant: str | None = None,
    seed: int | None = None,
) -> EvaluationWorkspace:
    """Build a workspace, optionally identifying the probe/mode variant."""
    return EvaluationWorkspace(config, variant=variant, seed=seed)


def get_layer_workspace(
    workspace: EvaluationWorkspace, layer: int
) -> EvaluationWorkspace:
    """Output workspace of one probed layer of *workspace*'s view."""
    return get_evaluation_workspace(
        workspace.config,
        variant=f"{workspace.variant}-layer_{layer}",
        seed=workspace.seed,
    )


def get_external_workspace(
    workspace: EvaluationWorkspace, external: Any
) -> EvaluationWorkspace:
    """Workspace of one external test set (``external`` has name and repo id)."""
    return get_evaluation_workspace(
        workspace.config.model_copy(
            update={
                "dataset_name": external.name,
                "dataset_repo_id": external.dataset_repo_id,
                "dataset_revision": external.dataset_revision,
                "dataset_subset": external.dataset_subset,
            }
        ),
        variant=workspace.variant,
        seed=workspace.seed,
    )
