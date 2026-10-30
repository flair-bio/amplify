"""On-disk layout for one (model, dataset) evaluation workspace and its run variants."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from modules.evaluate.src.contracts.config import (
    DatasetWorkspaceConfig,
    EvaluationWorkspaceConfig,
    validate_directory_label,
)
from modules.evaluate.src.tasks import TaskType
from modules.evaluate.src.utils.io import update_run_manifest, write_json

__all__ = [
    "DatasetWorkspaceConfig",
    "EvaluationWorkspace",
    "EvaluationWorkspaceConfig",
    "TaskType",
    "get_evaluation_workspace",
]


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
        self.base_path = config.base_path
        if variant is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", variant
        ):
            raise ValueError("Run variants must be safe single-directory labels.")
        workspace_root = config.dataset_root / config.model_name
        self.workspace_root = workspace_root
        temporary_root = (
            Path(config.tmp_path) if config.tmp_path else config.benchmark_root / "tmp"
        )
        self.tmp_path = (
            temporary_root / config.task_type / config.dataset_name / config.model_name
        )
        # Disposable: safe to delete between runs, PrepareStep recomputes on miss.
        # Deliberately NOT nested under `variant`: a frozen trunk's cached
        # embeddings don't depend on head_type/mode, so every variant
        # sweep for this model+dataset can reuse the same cache.
        self.embeddings_path = self.tmp_path / "embeddings"
        output_identity = hashlib.sha256(
            str(workspace_root.resolve()).encode()
        ).hexdigest()
        self.locks_path = self.tmp_path / "locks" / output_identity

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

    @contextmanager
    def reserve_run(self, *, overwrite: bool = False) -> Iterator[list[Path]]:
        """Hold a nonblocking lock until this ordinary pipeline finishes."""
        if self.variant is None:
            raise ValueError("Pipeline output protection requires a run variant.")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.locks_path.mkdir(parents=True, exist_ok=True)
        suffix = "" if self.seed is None else f"-seed_{self.seed}"
        lock_path = self.locks_path / f".{self.run_name}{suffix}.lock"
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError(
                    f"Evaluation is already running for {self.run_root}."
                ) from error
            runs_path = self.workspace_root
            if runs_path.is_symlink():
                raise ValueError(
                    f"Refusing to modify symlinked run outputs: {runs_path}"
                )
            dependent_pattern = re.compile(
                rf"{re.escape(self.variant)}-layer_\d+-{re.escape(self.config.run_id)}"
            )
            targets = [self.run_root]
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
            with (self.locks_path / ".workspace.lock").open("a") as workspace_lock:
                fcntl.flock(workspace_lock, fcntl.LOCK_EX)
                if self.seed is None and any(
                    self.experiment_root.glob("seed_*/run_manifest.json")
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
                for target in targets:
                    if target.exists() and any(target.iterdir()) and not overwrite:
                        raise FileExistsError(
                            f"Evaluation outputs already exist at {target}. "
                            "Choose another workspace.run_id or use --overwrite "
                            "to replace this variant and its derived outputs."
                        )
                for pattern in ("*/run_manifest.json", "*/seed_*/run_manifest.json"):
                    for path in self.workspace_root.glob(pattern):
                        provenance = json.loads(path.read_text()).get("workspace")
                        if provenance is not None and provenance != self._identity():
                            raise ValueError(
                                f"Output labels already identify different inputs at "
                                f"{self.workspace_root}; choose a new model/dataset label or benchmark root."
                            )
                if self.run_manifest_path.is_symlink():
                    raise ValueError("Refusing to modify symlinked run manifest")
                update_run_manifest(
                    self.run_manifest_path, "workspace", self._identity()
                )
            yield targets

    def _identity(self) -> dict[str, Any]:
        identity = {
            "dataset_name": self.dataset_name,
            "dataset_repo_id": self.dataset_repo_id,
            "task_type": self.task_type,
            "model_name": self.model_name,
            "model_repo_id": self.model_repo_id,
        }
        for key in ("dataset_config_name", "dataset_revision"):
            value = getattr(self.config, key)
            if value is not None:
                identity[key] = value
        return identity

    def initialize_run(
        self, targets: list[Path], *, overwrite: bool, config: dict[str, Any]
    ) -> None:
        """Clean only reserved variant outputs, preserving caches and siblings."""
        if self.seed is not None:
            identity = {key: value for key, value in config.items() if key != "seed"}
            self.validate_experiment("config", identity)
        experiment = (
            json.loads(self.run_manifest_path.read_text()).get("experiment", {})
            if self.run_manifest_path.exists()
            else {}
        )
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
                    if target != self.run_root:
                        shutil.rmtree(target)
                        continue
                    # Keep identity visible to concurrent sibling seed runs.
                    for child in target.iterdir():
                        if child == self.run_manifest_path:
                            continue
                        if child.is_dir() and not child.is_symlink():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
        manifest = {
            "workspace": self._identity(),
            "evaluation": {"status": "running", "config": config},
        }
        if experiment:
            manifest["experiment"] = experiment
        write_json(self.run_manifest_path, manifest)

    def validate_experiment(self, kind: str, identity: dict[str, Any]) -> None:
        """Bind sibling seed runs to identical settings and immutable inputs."""
        if self.experiment_root.is_symlink() or self.workspace_root.is_symlink():
            raise ValueError("Refusing to modify symlinked experiment outputs")
        validate_directory_label(kind)
        path = self.run_manifest_path
        self.locks_path.mkdir(parents=True, exist_ok=True)
        lock_path = self.locks_path / f".{self.run_name}-experiment.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_symlink():
                raise ValueError("Refusing to modify symlinked experiment identity")
            siblings = list(self.experiment_root.glob("seed_*/run_manifest.json"))
            if path not in siblings:
                siblings.append(path)
            for sibling in siblings:
                if not sibling.exists():
                    continue
                previous = json.loads(sibling.read_text()).get("experiment", {})
                if kind in previous and previous[kind] != identity:
                    raise ValueError(
                        f"Experiment {kind} differs across seeds; choose a new run_id"
                    )
            experiment = (
                json.loads(path.read_text()).get("experiment", {})
                if path.exists()
                else {}
            )
            update_run_manifest(path, "experiment", {**experiment, kind: identity})


def get_evaluation_workspace(
    config: EvaluationWorkspaceConfig,
    variant: str | None = None,
    seed: int | None = None,
) -> EvaluationWorkspace:
    """Build a workspace, optionally identifying the probe/mode variant."""
    return EvaluationWorkspace(config, variant=variant, seed=seed)
