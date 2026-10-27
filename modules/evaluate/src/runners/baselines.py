"""Explicit, model-free sequence baseline runs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import DatasetDict
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.resolution import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.dataset.workspace import DatasetWorkspaceConfig
from modules.evaluate.src.steps.prepare import PrepareConfig
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.baselines import (
    BaselineName,
    BaselinesConfig,
    artifact_directory,
    artifact_identity,
    baseline_context,
    build_sequence_baseline,
    external_baseline_context,
    load_baseline_artifact,
    sequence_rows,
)
from modules.evaluate.src.utils.io import log_to_file, write_json

logger = logging.getLogger(__name__)


class BaselineRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: DatasetWorkspaceConfig
    prepare: PrepareConfig = Field(
        default_factory=lambda: PrepareConfig.model_validate({})
    )
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)
    external_test_sets: list[ExternalTestSetConfig] = Field(default_factory=list)
    seed: int = 710019

    @model_validator(mode="after")
    def validate_baselines(self) -> BaselineRunConfig:
        self.baselines.validate_task(self.workspace.task_type)
        if not self.baselines.names:
            raise ValueError("Select at least one baseline in baselines.names.")
        names = [external.name for external in self.external_test_sets]
        if len(set(names)) != len(names):
            raise ValueError("external_test_sets names must be unique.")
        task = get_task(self.workspace.task_type)
        task.validate_dataset_config(
            self.workspace.target_type,
            self.workspace.label_format,
            self.workspace.dataset_name,
        )
        return self


def load_baseline_config(
    paths: Sequence[str | Path], overrides: Sequence[str] | None = None
) -> BaselineRunConfig:
    from modules.evaluate.src.utils.config_loader import compose_evaluation_config

    raw: Any = OmegaConf.to_container(
        compose_evaluation_config(paths, overrides), resolve=True
    )
    if raw.get("cross_validation", {}).get("enabled"):
        raise ValueError("Standalone baselines currently support non-CV runs only.")
    # Model and tuning settings do not apply to the baseline route.
    workspace = {
        key: value
        for key, value in raw["workspace"].items()
        if key in DatasetWorkspaceConfig.model_fields
    }
    return BaselineRunConfig.model_validate(
        {
            "workspace": workspace,
            "prepare": raw.get("steps", {}).get("prepare", {}),
            "baselines": raw.get("baselines", {}),
            "external_test_sets": raw.get("external_test_sets", []),
            "seed": raw.get("seed", 710019),
        }
    )


def run_baselines(
    config: BaselineRunConfig, dataset: DatasetDict | None = None
) -> dict[str, dict[str, Path]]:
    """Build train-only baselines for the main and external test datasets."""
    workspace, prepare = config.workspace, config.prepare
    data = resolve_evaluation_data(
        workspace.dataset_repo_id or workspace.dataset_name,
        split=prepare.split,
        seed=config.seed,
        validation_fraction=prepare.validation_fraction,
        dataset=dataset,
    )
    context = baseline_context(
        workspace,
        data,
        config.baselines.names,
        sequence_column=prepare.sequence_column,
        label_column=prepare.label_column,
        id_column=prepare.id_column,
        split_roles=prepare.split.model_dump(),
    )
    row_options: dict[str, Any] = {
        "sequence_column": prepare.sequence_column,
        "label_column": prepare.label_column,
        "id_column": prepare.id_column,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
    }
    train_rows = sequence_rows(data["train"], **row_options)
    tests: list[tuple[str, dict[str, Any], DatasetDict]] = [("test", context, data)]
    for external in config.external_test_sets:
        external_data = resolve_external_data(external.dataset_repo_id, external.split)
        external_context = external_baseline_context(
            context,
            workspace,
            external_data,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            sequence_column=prepare.sequence_column,
            label_column=prepare.label_column,
            id_column=prepare.id_column,
        )
        tests.append((f"external/{external.name}", external_context, external_data))

    outputs: dict[str, dict[str, Path]] = {}
    for test_name, test_context, test_data in tests:
        outputs[test_name] = {}
        test_rows = sequence_rows(test_data["test"], **row_options)
        for name in config.baselines.names:
            directory = artifact_directory(test_context, name)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory.parent / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                fit_path = directory.parent / "fit_manifest.json"
                fit_identity = {"baseline": name, "fit": test_context["fit"]}
                if fit_path.exists():
                    if json.loads(fit_path.read_text()) != fit_identity:
                        raise ValueError(
                            f"Incompatible baseline fit manifest: {fit_path}"
                        )
                else:
                    write_json(fit_path, fit_identity)
                with log_to_file(directory / "log"):
                    logger.info(
                        "Resolving baseline %s for %s at %s", name, test_name, directory
                    )
                    _publish_baseline(
                        directory,
                        test_context,
                        name,
                        train_rows,
                        test_rows,
                        workspace.num_labels,
                    )
                    _, path = load_baseline_artifact(test_context, name)
                    logger.info("Baseline predictions ready: %s", path)
                outputs[test_name][name] = path
    return outputs


def _publish_baseline(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_labels: int | None,
) -> None:
    if not (directory / "manifest.json").exists():
        frame = build_sequence_baseline(name, train_rows, test_rows, num_labels)
        temporary = directory / f".{uuid4().hex}.parquet"
        path = directory / "predictions.parquet"
        try:
            frame.write_parquet(temporary)
            temporary.replace(path)
            write_json(
                directory / "manifest.json",
                {
                    "identity": artifact_identity(context, name),
                    "rows": frame.height,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)
