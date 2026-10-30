"""Generate and publish model-independent baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from datasets import DatasetDict

from modules.evaluate.src.contracts.config import (
    BaselineName,
    BaselineRunConfig,
)
from modules.evaluate.src.dataset.baseline_data import (
    artifact_directory,
    artifact_identity,
    baseline_context,
    contact_baseline_context,
    external_baseline_context,
    load_baseline_artifact,
    mmseqs_coverage,
    sequence_rows,
    unlabeled_sequence_rows,
)
from modules.evaluate.src.dataset.contact import (
    load_contact_dataset,
    prepare_contact_dataset,
    separation_prior_prediction_rows,
)
from modules.evaluate.src.dataset.dataloader import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.dataset.subsets import select_test_subsets
from modules.evaluate.src.model.baselines import (
    build_amino_acid_frequency_baseline,
    build_sequence_baseline,
)
from modules.evaluate.src.utils.io import log_to_file, write_json
from modules.evaluate.src.utils.storage import cache_lock

logger = logging.getLogger(__name__)


def run_baselines(
    config: BaselineRunConfig, dataset: DatasetDict | None = None
) -> dict[str, dict[str, Path]]:
    """Build shared baselines for the main and external test datasets."""
    workspace, prepare = config.workspace, config.prepare
    if config.test_subsets:
        data = resolve_evaluation_data(
            workspace.dataset_repo_id or workspace.dataset_name,
            split=prepare.split,
            seed=config.seed,
            validation_fraction=prepare.validation_fraction,
            dataset=dataset,
            config_name=workspace.dataset_config_name,
            revision=workspace.dataset_revision,
        )
        if workspace.task_type == "contact_prediction":
            prepare_contact_dataset(data)
        subsets = select_test_subsets(
            data,
            config.test_subsets,
            group_column="record_id"
            if workspace.task_type == "contact_prediction"
            else None,
        )
        subset_results: dict[str, dict[str, Path]] = {}
        for index, (name, subset) in enumerate(subsets.items()):
            subset_outputs = run_baselines(
                config.model_copy(
                    update={
                        "test_subsets": {},
                        "external_test_sets": config.external_test_sets
                        if index == 0
                        else [],
                    }
                ),
                dataset=subset,
            )
            subset_results[f"tests/{name}"] = subset_outputs.pop("test")
            subset_results.update(subset_outputs)
        return subset_results
    if workspace.task_type == "contact_prediction":
        if config.external_test_sets:
            raise ValueError(
                "Contact separation_prior does not support external test sets."
            )
        return _run_contact_baselines(config, dataset)

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
        max_length=prepare.max_length,
    )
    row_options: dict[str, Any] = {
        "sequence_column": prepare.sequence_column,
        "label_column": prepare.label_column,
        "id_column": prepare.id_column,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
    }
    if workspace.task_type == "pseudo_perplexity":
        train_rows: list[dict[str, Any]] = []
        test_rows = unlabeled_sequence_rows(
            data["test"],
            sequence_column=prepare.sequence_column,
            id_column=prepare.id_column,
        )
    else:
        train_rows = sequence_rows(data["train"], **row_options)
        test_rows = sequence_rows(data["test"], **row_options)
    tests: list[tuple[str, dict[str, Any], DatasetDict, list[dict[str, Any]]]] = [
        ("test", context, data, test_rows)
    ]
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
            max_length=prepare.max_length,
        )
        external_rows = (
            unlabeled_sequence_rows(
                external_data["test"],
                sequence_column=prepare.sequence_column,
                id_column=prepare.id_column,
            )
            if workspace.task_type == "pseudo_perplexity"
            else sequence_rows(external_data["test"], **row_options)
        )
        tests.append(
            (
                f"external/{external.name}",
                external_context,
                external_data,
                external_rows,
            )
        )

    outputs: dict[str, dict[str, Path]] = {}
    for test_name, test_context, _, test_rows in tests:
        outputs[test_name] = {}
        for name in config.baselines.names:
            directory = artifact_directory(test_context, name)
            directory.mkdir(parents=True, exist_ok=True)
            with cache_lock(directory.parent / ".lock"):
                fit_path = directory.parent / "fit_manifest.json"
                fit_identity = {
                    "baseline": name,
                    "fit": artifact_identity(test_context, name)["fit"],
                }
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
                        task_type=workspace.task_type,
                        seed=config.seed,
                        max_residues=prepare.max_length - 2,
                    )
                    _, path = load_baseline_artifact(test_context, name)
                    logger.info("Baseline predictions ready: %s", path)
                outputs[test_name][name] = path
    return outputs


def _run_contact_baselines(
    config: BaselineRunConfig, dataset: DatasetDict | None
) -> dict[str, dict[str, Path]]:
    workspace = config.workspace
    source = workspace.dataset_repo_id or workspace.dataset_name
    contacts = (
        load_contact_dataset(
            source,
            revision=workspace.dataset_revision,
            config_name=workspace.dataset_config_name,
        )
        if dataset is None
        else prepare_contact_dataset(
            dataset,
            source=source,
            revision=workspace.dataset_revision,
        )
    )
    context = contact_baseline_context(workspace, contacts, config.baselines.names)
    rows = separation_prior_prediction_rows(contacts)
    if not rows:
        raise ValueError("Contact separation_prior requires test scoring rows.")
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("original_i", "original_j").cast(pl.List(pl.Int32)),
        pl.col("labels").cast(pl.List(pl.Int8)),
        pl.col("scores").cast(pl.List(pl.Float64)),
    )
    outputs: dict[str, dict[str, Path]] = {"test": {}}
    for name in config.baselines.names:
        directory = artifact_directory(context, name)
        directory.mkdir(parents=True, exist_ok=True)
        with cache_lock(directory.parent / ".lock"):
            fit_path = directory.parent / "fit_manifest.json"
            fit_identity = {"baseline": name, "fit": context["fit"]}
            if fit_path.exists():
                if json.loads(fit_path.read_text()) != fit_identity:
                    raise ValueError(f"Incompatible baseline fit manifest: {fit_path}")
            else:
                write_json(fit_path, fit_identity)
            _publish_contact_baseline(directory, context, name, frame)
            _, path = load_baseline_artifact(context, name)
            outputs["test"][name] = path
    return outputs


def _publish_contact_baseline(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    frame: pl.DataFrame,
) -> None:
    if (directory / "manifest.json").exists():
        return
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


def _publish_baseline(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_labels: int | None,
    *,
    task_type: str,
    seed: int,
    max_residues: int,
) -> None:
    if not (directory / "manifest.json").exists():
        frame = (
            build_amino_acid_frequency_baseline(test_rows, max_residues=max_residues)
            if name == "amino_acid_frequency"
            else build_sequence_baseline(
                name,
                task_type,
                train_rows,
                test_rows,
                num_labels,
                seed=seed,
            )
        )
        temporary = directory / f".{uuid4().hex}.parquet"
        path = directory / "predictions.parquet"
        try:
            metadata = {}
            if name == "mmseqs_knn":
                coverage = mmseqs_coverage(frame)
                metadata["coverage"] = coverage
                logger.info("MMseqs2 significant-hit coverage: %s", coverage)
            frame.write_parquet(temporary)
            temporary.replace(path)
            write_json(
                directory / "manifest.json",
                {
                    "identity": artifact_identity(context, name),
                    "rows": frame.height,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    **metadata,
                },
            )
        finally:
            temporary.unlink(missing_ok=True)
