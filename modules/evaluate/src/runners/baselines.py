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

from modules.evaluate.src.config.score import ScoreConfig
from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.preparation import PrepareConfig
from modules.evaluate.src.dataset.resolution import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.dataset.workspace import DatasetWorkspaceConfig
from modules.evaluate.src.schemas.artifacts import EvaluationMetadata, PredictOutput
from modules.evaluate.src.scoring.compute import bootstrap_scores, score_conditions
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.baselines import (
    STRUCTURAL_TASK_BASELINES,
    BaselineName,
    BaselinesConfig,
    artifact_directory,
    artifact_identity,
    baseline_context,
    build_sequence_baseline,
    external_baseline_context,
    identity_hash,
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
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    external_test_sets: list[ExternalTestSetConfig] = Field(default_factory=list)
    seed: int = 710019

    @model_validator(mode="after")
    def validate_baselines(self) -> BaselineRunConfig:
        self.baselines.validate_task(self.workspace.task_type)
        if not self.baselines.names:
            raise ValueError("Select at least one baseline in baselines.names.")
        if self.workspace.task_type in STRUCTURAL_TASK_BASELINES and (
            self.external_test_sets
        ):
            raise ValueError(
                f"external_test_sets are not supported for "
                f"task_type='{self.workspace.task_type}' baselines."
            )
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
            "score": raw.get("steps", {}).get("score", {}),
            "external_test_sets": raw.get("external_test_sets", []),
            "seed": raw.get("seed", 710019),
        }
    )


def run_baselines(
    config: BaselineRunConfig, dataset: DatasetDict | None = None
) -> dict[str, dict[str, Path]]:
    """Build train-only baselines for the main and external test datasets."""
    if config.workspace.task_type in STRUCTURAL_TASK_BASELINES:
        return {"test": _run_structural_baselines(config)}
    workspace, prepare = config.workspace, config.prepare
    data = resolve_evaluation_data(
        workspace.dataset_repo_id or workspace.dataset_name,
        split=prepare.split,
        seed=config.seed,
        validation_fraction=prepare.validation_fraction,
        dataset=dataset,
        revision=workspace.dataset_revision,
        subset=workspace.dataset_subset,
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
            revision=external.dataset_revision,
            subset=external.dataset_subset,
        )
        external_context = external_baseline_context(
            context,
            workspace,
            external_data,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            sequence_column=prepare.sequence_column,
            label_column=prepare.label_column,
            id_column=prepare.id_column,
            dataset_revision=external.dataset_revision,
            dataset_subset=external.dataset_subset,
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
                    frame, path = load_baseline_artifact(test_context, name)
                    _score_baseline(config, test_context, name, frame, path)
                    logger.info("Baseline predictions ready: %s", path)
                outputs[test_name][name] = path
    return outputs


def _score_baseline(
    config: BaselineRunConfig,
    context: dict[str, Any],
    name: BaselineName,
    frame: Any,
    predictions_path: Path,
) -> None:
    task = get_task(config.workspace.task_type, config.workspace.num_labels)
    metric_names = config.score.metrics or task.default_metrics()
    protocol = {
        "config": config.score.model_dump(mode="json"),
        "metrics": metric_names,
        "seed": config.seed,
    }
    identity = artifact_identity(context, name)
    checksum = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    path = predictions_path.parent / "scores" / f"{identity_hash(protocol)}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved.get("identity") != identity
            or saved.get("protocol") != protocol
            or saved.get("predictions_sha256") != checksum
        ):
            raise ValueError(f"Incompatible standalone baseline scores: {path}")
        return
    output = PredictOutput(
        metadata=EvaluationMetadata(
            model_name=name,
            dataset_name=config.workspace.dataset_name,
            task_type=config.workspace.task_type,
        ),
        predictions=frame,
    )
    scored = score_conditions(
        task, output, metric_names, config.score.metric_aggregation, config.seed
    )
    payload: dict[str, Any] = {
        "identity": identity,
        "protocol": protocol,
        "predictions_sha256": checksum,
        "population_sha256": identity_hash(
            frame.select(
                *[
                    column
                    for column in ("id", "label", "evaluated_positions")
                    if column in frame.columns
                ]
            ).to_dicts()
        ),
        "scores": scored.trained,
    }
    if config.score.bootstrap_enabled:
        bootstrap = bootstrap_scores(
            task,
            output,
            metric_names,
            config.score.metric_aggregation,
            n_samples=config.score.n_samples,
            confidence_interval=config.score.confidence_interval,
            n_permutations=config.score.n_permutations,
            seed=config.seed,
            trained_per_example_scores=scored.trained_per_example,
        )
        payload["confidence_intervals"] = bootstrap.confidence_intervals
    write_json(path, payload)


def _run_structural_baselines(config: BaselineRunConfig) -> dict[str, Path]:
    """Baseline-only runs for contact (separation prior) and PPL (AA frequency).

    Structural tasks have no generic train/test row mapping, so these bypass the
    sequence path. Network access happens only here, at execution.
    """
    from modules.evaluate.src.artifacts.identity import dataset_identity
    from modules.evaluate.src.runners.structural_baselines import (
        AMINO_ACID_BASELINE,
        CONTACT_BASELINE,
        load_amino_acid_reference,
        run_amino_acid_frequency_baseline,
        run_contact_separation_baseline,
    )
    from modules.evaluate.src.tasks.contact_prediction import ContactPredictionTask

    workspace, prepare = config.workspace, config.prepare
    source = workspace.dataset_repo_id or workspace.dataset_name
    root = workspace.dataset_root / "baselines"
    identity = dataset_identity(
        source=source,
        subset=workspace.dataset_subset,
        revision=workspace.dataset_revision,
    )
    outputs: dict[str, Path] = {}
    if workspace.task_type == "contact_prediction":
        if config.baselines.names != [CONTACT_BASELINE]:
            raise ValueError(f"contact_prediction supports only {CONTACT_BASELINE}")
        dataset = ContactPredictionTask().prepare_dataset(
            source, revision=workspace.dataset_revision, edition=workspace.edition
        )
        outputs[CONTACT_BASELINE] = run_contact_separation_baseline(
            dataset=dataset,
            dataset_identity=identity,
            root=root,
            seed=config.seed,
        )
    elif workspace.task_type == "pseudo_perplexity":
        if config.baselines.names != [AMINO_ACID_BASELINE]:
            raise ValueError(f"pseudo_perplexity supports only {AMINO_ACID_BASELINE}")
        reference = config.baselines.amino_acid_reference
        if reference is None:
            raise ValueError(
                "amino_acid_frequency requires baselines.amino_acid_reference "
                "(a pinned train-only reference corpus)."
            )
        reference_sequences = load_amino_acid_reference(reference)
        data = resolve_external_data(
            source,
            "test",
            revision=workspace.dataset_revision,
            subset=workspace.dataset_subset,
        )["test"]
        test_rows = [
            (str(row[prepare.id_column]), str(row[prepare.sequence_column]))
            for row in data
        ]
        outputs[AMINO_ACID_BASELINE] = run_amino_acid_frequency_baseline(
            reference=reference,
            reference_sequences=reference_sequences,
            test_rows=test_rows,
            dataset_identity=identity,
            root=root,
        )
    else:  # pragma: no cover - guarded by STRUCTURAL_TASK_BASELINES
        raise ValueError(f"No structural baseline for {workspace.task_type}")
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
