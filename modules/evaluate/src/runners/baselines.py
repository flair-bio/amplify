"""Generate and publish model-independent sequence baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from datasets import DatasetDict

from modules.evaluate.src.contracts.config import (
    BaselineName,
    BaselineRunConfig,
)
from modules.evaluate.src.dataset.baseline_data import (
    artifact_directory,
    artifact_identity,
    baseline_context,
    external_baseline_context,
    load_baseline_artifact,
    sequence_rows,
)
from modules.evaluate.src.dataset.dataloader import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.model.baselines import build_sequence_baseline
from modules.evaluate.src.utils.io import log_to_file, write_json
from modules.evaluate.src.utils.storage import cache_lock

logger = logging.getLogger(__name__)


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
        config_name=workspace.dataset_config_name,
        revision=workspace.dataset_revision,
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
        external_data = resolve_external_data(
            external.dataset_repo_id,
            external.split,
            config_name=external.dataset_config_name,
            revision=external.dataset_revision,
        )
        external_context = external_baseline_context(
            context,
            workspace,
            external_data,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            config_name=external.dataset_config_name,
            revision=external.dataset_revision,
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
            with cache_lock(directory.parent / ".lock"):
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
