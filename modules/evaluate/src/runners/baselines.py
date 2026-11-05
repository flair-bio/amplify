"""Generate and publish model-independent baseline artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl
from datasets import DatasetDict

from modules.evaluate.src.contracts.artifacts import (
    ContactEvaluationMetadata,
    EvaluationMetadata,
    PredictOutput,
    StructuredScoreOutput,
)
from modules.evaluate.src.contracts.config import (
    BaselineName,
    BaselineRunConfig,
    ScoreConfig,
)
from modules.evaluate.src.dataset.baseline_data import (
    artifact_directory,
    artifact_identity,
    baseline_context,
    contact_baseline_context,
    external_baseline_context,
    identity_hash,
    load_baseline_artifact,
    mmseqs_coverage,
    preprocessing_identity,
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
    _mmseqs_nearest_neighbors,
    build_amino_acid_frequency_baseline,
    build_sequence_baseline,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.io import write_json, write_parquet
from modules.evaluate.src.utils.storage import cache_lock

logger = logging.getLogger(__name__)


def run_baselines(
    config: BaselineRunConfig,
    dataset: DatasetDict | None = None,
    *,
    _mmseqs_neighbor_cache: dict[str, list[int | None]] | None = None,
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
                _mmseqs_neighbor_cache=_mmseqs_neighbor_cache,
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
        preprocessing=preprocessing_identity(
            sequence_column=prepare.sequence_column,
            label_column=prepare.label_column,
            id_column=prepare.id_column,
            split_roles=prepare.split.model_dump(),
            max_length=prepare.max_length,
            random_truncate_by_split=prepare.random_truncate_by_split,
            seed=config.seed,
        ),
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
            preprocessing=preprocessing_identity(
                sequence_column=prepare.sequence_column,
                label_column=prepare.label_column,
                id_column=prepare.id_column,
                split_roles={"name": external.name, "test": external.split},
                max_length=prepare.max_length,
                random_truncate_by_split=prepare.random_truncate_by_split,
                seed=config.seed,
            ),
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
                _upgrade_legacy_baseline(
                    directory, test_context, name, config, test_name
                )
                mmseqs_neighbors = None
                if (
                    name == "mmseqs_knn"
                    and _mmseqs_neighbor_cache is not None
                    and not (directory / "run_manifest.json").exists()
                ):
                    sequence_identity = identity_hash(
                        {
                            "train": [row["sequence"] for row in train_rows],
                            "test": [row["sequence"] for row in test_rows],
                        }
                    )
                    if sequence_identity not in _mmseqs_neighbor_cache:
                        _mmseqs_neighbor_cache[sequence_identity] = (
                            _mmseqs_nearest_neighbors(train_rows, test_rows)
                        )
                    mmseqs_neighbors = _mmseqs_neighbor_cache[sequence_identity]
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
                    mmseqs_neighbors=mmseqs_neighbors,
                )
                frame, path = load_baseline_artifact(test_context, name)
                _publish_standalone_scores(
                    directory,
                    test_context,
                    name,
                    frame,
                    config.score,
                    workspace,
                    config.seed,
                    run_config=config.model_dump(mode="json"),
                    test_name=test_name,
                )
                logger.info("Baseline predictions ready: %s", path)
                outputs[test_name][name] = path
    return outputs


def run_multi_target_baselines(
    config: BaselineRunConfig, dataset: DatasetDict | None = None
) -> dict[str, dict[str, Path]]:
    """Run scalar baselines per target and publish an unweighted metric mean."""
    multi_target = config.multi_target
    if multi_target is None:
        raise ValueError("Multi-target baselines require multi_target configuration.")
    if config.external_test_sets or config.test_subsets:
        raise ValueError(
            "Multi-target baselines do not support external test sets or test subsets."
        )
    task = get_task(config.workspace.task_type)
    if multi_target.metric not in task.metrics:
        raise ValueError(
            f"Metric {multi_target.metric!r} is not supported for "
            f"task_type={config.workspace.task_type!r}."
        )

    scores_config = config.score
    if (
        scores_config.metrics is not None
        and multi_target.metric not in scores_config.metrics
    ):
        scores_config = scores_config.model_copy(
            update={"metrics": [*scores_config.metrics, multi_target.metric]}
        )

    property_outputs: dict[str, dict[str, Path]] = {}
    mmseqs_neighbor_cache: dict[str, list[int | None]] = {}
    for property_name in multi_target.target_columns:
        property_config = config.model_copy(
            update={
                "multi_target": None,
                "prepare": config.prepare.model_copy(
                    update={"label_column": property_name}
                ),
                "score": scores_config,
            }
        )
        outputs = run_baselines(
            property_config,
            dataset=dataset,
            _mmseqs_neighbor_cache=mmseqs_neighbor_cache,
        )
        if set(outputs) != {"test"}:
            raise ValueError(
                "Multi-target baseline aggregation requires exactly one test output."
            )
        property_outputs[property_name] = outputs["test"]

    aggregate_outputs: dict[str, Path] = {}
    for name in config.baselines.names:
        rows = []
        target_identities = {}
        for property_name in multi_target.target_columns:
            try:
                prediction_path = property_outputs[property_name][name]
            except KeyError as error:
                raise ValueError(
                    f"Missing {name} baseline output for property {property_name!r}."
                ) from error
            manifest_path = prediction_path.with_name("run_manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                raise ValueError(
                    f"Incomplete {name} baseline run for {property_name!r}: "
                    f"{manifest_path}"
                )
            target_identity = manifest.get("identity")
            if not isinstance(target_identity, dict):
                raise TypeError(f"Invalid baseline identity: {manifest_path}")
            target_identities[property_name] = target_identity
            value = _baseline_metric_value(
                manifest.get("score", {}).get("scores"), multi_target.metric
            )
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"Invalid {multi_target.metric} for {name} baseline "
                    f"and property {property_name!r}."
                )
            rows.append(
                {
                    multi_target.target_key: property_name,
                    multi_target.metric: float(value),
                    "manifest_path": str(manifest_path),
                }
            )

        table = pl.DataFrame(rows)
        summary = {
            "metadata": {
                "model_name": "baselines",
                "dataset_name": config.workspace.dataset_name,
                "dataset_repo_id": config.workspace.dataset_repo_id,
                "dataset_config_name": config.workspace.dataset_config_name,
                "dataset_revision": config.workspace.dataset_revision,
                "task_type": config.workspace.task_type,
                "split": config.prepare.split.name,
                "seed": config.seed,
                "baseline": name,
            },
            "run_name": f"baselines-{name}",
            "metric": multi_target.metric,
            "aggregation": multi_target.aggregation_name,
            multi_target.count_key: len(multi_target.target_columns),
            "value": math.fsum(table[multi_target.metric].to_list())
            / len(multi_target.target_columns),
        }
        identity = {
            "config": config.model_dump(mode="json"),
            "baseline": name,
            "targets": target_identities,
        }
        output_root = (
            config.workspace.dataset_root
            / "baselines"
            / "multi_target"
            / name
            / f"run-{identity_hash(identity)[:16]}"
        )
        table_path = output_root / multi_target.score_table_filename
        manifest_path = output_root / "run_manifest.json"
        with cache_lock(output_root.parent / f".{output_root.name}.lock"):
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                if previous.get("evaluation", {}).get("config") != identity["config"]:
                    raise ValueError(
                        f"Incompatible multi-target baseline manifest: {manifest_path}"
                    )
            write_json(
                manifest_path,
                {
                    "workspace": config.workspace.model_dump(mode="json"),
                    "evaluation": {
                        "status": "running",
                        "config": identity["config"],
                    },
                },
            )
            write_parquet(table, table_path)
            write_json(output_root / "summary.json", summary)
            write_json(
                manifest_path,
                {
                    "workspace": config.workspace.model_dump(mode="json"),
                    "evaluation": {
                        "status": "complete",
                        "config": identity["config"],
                    },
                    "score": summary,
                },
            )
        aggregate_outputs[name] = table_path
        logger.info(
            "Multi-target baseline %s mean %s over %d properties: %.6g",
            name,
            multi_target.metric,
            len(multi_target.target_columns),
            summary["value"],
        )
    return {"test": aggregate_outputs}


def _baseline_metric_value(scores: Any, metric: str) -> Any:
    if isinstance(scores, dict):
        return scores.get(metric)
    if isinstance(scores, list):
        for row in scores:
            if isinstance(row, dict) and row.get("metric") == metric:
                return row.get("value")
    return None


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
    effective_score_config = _baseline_score_config(config.score)
    for name in config.baselines.names:
        directory = artifact_directory(context, name)
        directory.mkdir(parents=True, exist_ok=True)
        with cache_lock(directory.parent / ".lock"):
            _upgrade_legacy_baseline(directory, context, name, config, "test")
            _publish_contact_baseline(directory, context, name, frame)
            artifact, path = load_baseline_artifact(context, name)
            metadata = ContactEvaluationMetadata(
                model_name="baselines",
                dataset_name=workspace.dataset_name,
                dataset_repo_id=workspace.dataset_repo_id,
                dataset_config_name=workspace.dataset_config_name,
                dataset_revision=contacts.revision,
                task_type=workspace.task_type,
                seed=config.seed,
                dataset_fingerprint=contacts.fingerprint,
                schema_version=contacts.schema_version,
                protocol_version="casp-contact-v1",
            )
            result = get_task(workspace.task_type).score(
                PredictOutput(metadata=metadata, predictions=artifact),
                effective_score_config,
                seed=config.seed,
            )
            if not isinstance(result, StructuredScoreOutput):
                raise TypeError(
                    "Contact baseline scoring must return structured scores."
                )
            scores_path = write_parquet(
                result.summaries.with_columns(pl.lit(name).alias("baseline")),
                directory / "scores.parquet",
            )
            write_json(
                directory / "run_manifest.json",
                {
                    "status": "complete",
                    "identity": artifact_identity(context, name),
                    "artifact": _prediction_metadata(directory, artifact),
                    "run": {
                        "seed": config.seed,
                        "test": "test",
                        "config": _baseline_run_config(config, effective_score_config),
                    },
                    "score": {
                        "config": effective_score_config.model_dump(mode="json"),
                        "bootstrap": _baseline_bootstrap_status(),
                        "scores": result.summaries.to_dicts(),
                        "scores_path": str(scores_path),
                    },
                },
            )
            _remove_legacy_sidecars(directory)
            outputs["test"][name] = path
    return outputs


def _publish_contact_baseline(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    frame: pl.DataFrame,
) -> None:
    if (directory / "run_manifest.json").exists():
        return
    temporary = directory / f".{uuid4().hex}.parquet"
    path = directory / "predictions.parquet"
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
        write_json(
            directory / "run_manifest.json",
            {
                "status": "predictions_published",
                "identity": artifact_identity(context, name),
                "artifact": _prediction_metadata(directory, frame),
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
    mmseqs_neighbors: list[int | None] | None = None,
) -> None:
    if not (directory / "run_manifest.json").exists():
        if name == "amino_acid_frequency":
            frame = build_amino_acid_frequency_baseline(
                test_rows, max_residues=max_residues
            )
        elif mmseqs_neighbors is None:
            frame = build_sequence_baseline(
                name,
                task_type,
                train_rows,
                test_rows,
                num_labels,
                seed=seed,
            )
        else:
            frame = build_sequence_baseline(
                name,
                task_type,
                train_rows,
                test_rows,
                num_labels,
                seed=seed,
                mmseqs_neighbors=mmseqs_neighbors,
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
                directory / "run_manifest.json",
                {
                    "status": "predictions_published",
                    "identity": artifact_identity(context, name),
                    "artifact": _prediction_metadata(directory, frame, **metadata),
                },
            )
        finally:
            temporary.unlink(missing_ok=True)


def _publish_standalone_scores(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    artifact: pl.DataFrame,
    config: ScoreConfig,
    workspace: Any,
    seed: int,
    *,
    run_config: dict[str, Any],
    test_name: str,
) -> None:
    effective_score_config = _baseline_score_config(config)
    task = get_task(workspace.task_type, workspace.num_labels)
    if workspace.task_type == "token_classification":
        rows = []
        for row in artifact.to_dicts():
            valid_labels = [label for label in row["label"] if label != -100]
            item: dict[str, Any] = {
                "id": row["id"],
                "label": valid_labels,
                "prediction": [row["prediction"]] * len(valid_labels),
            }
            if "probability" in row:
                item["probability"] = [row["probability"]] * len(valid_labels)
            rows.append(item)
        artifact = pl.DataFrame(rows)
    elif "label" not in artifact.columns:
        artifact = artifact.with_columns(pl.lit(0).alias("label"))

    metadata = EvaluationMetadata(
        model_name="baselines",
        dataset_name=workspace.dataset_name,
        dataset_repo_id=workspace.dataset_repo_id,
        dataset_config_name=workspace.dataset_config_name,
        dataset_revision=workspace.dataset_revision,
        task_type=workspace.task_type,
        split=str(context["test"]["split"]),
        seed=seed,
    )
    result = task.score(
        PredictOutput(metadata=metadata, predictions=artifact),
        effective_score_config,
        seed=seed,
    )
    scores = (
        result.summaries.to_dicts() if hasattr(result, "summaries") else result.scores
    )
    if hasattr(result, "summaries"):
        score_frame = result.summaries.with_columns(pl.lit(name).alias("baseline"))
    else:
        if result.long_df is None:
            raise ValueError("Flat baseline scoring did not produce a score table.")
        score_frame = result.long_df.with_columns(pl.lit(name).alias("baseline"))
    scores_path = write_parquet(score_frame, directory / "scores.parquet")
    run_manifest_path = directory / "run_manifest.json"
    manifest = json.loads(run_manifest_path.read_text())
    manifest.update(
        {
            "status": "complete",
            "run": {
                "seed": seed,
                "test": test_name,
                "config": _baseline_run_config(run_config, effective_score_config),
            },
            "score": {
                "config": effective_score_config.model_dump(mode="json"),
                "bootstrap": _baseline_bootstrap_status(),
                "scores": scores,
                "scores_path": str(scores_path),
            },
        }
    )
    write_json(run_manifest_path, manifest)
    _remove_legacy_sidecars(directory)


def _baseline_score_config(config: ScoreConfig) -> ScoreConfig:
    """Return the settings actually used for standalone baseline scores."""
    return config.model_copy(update={"bootstrap_enabled": False})


def _baseline_run_config(
    config: BaselineRunConfig | dict[str, Any], score_config: ScoreConfig
) -> dict[str, Any]:
    """Persist effective scoring settings instead of the requested settings."""
    run_config = (
        config.model_dump(mode="json")
        if isinstance(config, BaselineRunConfig)
        else config
    )
    run_config["score"] = score_config.model_dump(mode="json")
    return run_config


def _baseline_bootstrap_status() -> dict[str, str]:
    return {
        "status": "not_computed",
        "reason": "Standalone baseline scoring reports point estimates only.",
    }


def _upgrade_legacy_baseline(
    directory: Path,
    context: dict[str, Any],
    name: BaselineName,
    config: BaselineRunConfig,
    test_name: str,
) -> None:
    """Consolidate a verified legacy artifact's JSON sidecars in place."""
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists():
        return
    legacy_manifest_path = directory / "manifest.json"
    if not legacy_manifest_path.exists():
        return

    identity = artifact_identity(context, name)
    legacy = json.loads(legacy_manifest_path.read_text())
    if legacy.get("identity") != identity:
        raise ValueError(f"Incompatible baseline manifest: {legacy_manifest_path}")
    fit_path = directory.parent / "fit_manifest.json"
    if fit_path.exists():
        fit_manifest = json.loads(fit_path.read_text())
        expected_fit = {"baseline": name, "fit": identity["fit"]}
        if fit_manifest != expected_fit:
            raise ValueError(f"Incompatible baseline fit manifest: {fit_path}")

    prediction_path = directory / "predictions.parquet"
    if not prediction_path.exists() or hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest() != legacy.get("sha256"):
        raise ValueError(f"Corrupt baseline predictions: {prediction_path}")

    artifact: dict[str, Any] = {
        "path": prediction_path.name,
        "rows": legacy.get("rows"),
        "sha256": legacy["sha256"],
    }
    if "coverage" in legacy:
        artifact["coverage"] = legacy["coverage"]
    score_path = directory / "scores.json"
    if score_path.exists():
        score_manifest = json.loads(score_path.read_text())
        if score_manifest.get("identity") != identity:
            raise ValueError(f"Incompatible baseline scores: {score_path}")
    write_json(
        manifest_path,
        {
            "status": "predictions_published",
            "identity": identity,
            "artifact": artifact,
            "run": {
                "seed": config.seed,
                "test": test_name,
                "config": config.model_dump(mode="json"),
            },
        },
    )


def _remove_legacy_sidecars(directory: Path) -> None:
    for path in (
        directory / "manifest.json",
        directory / "scores.json",
        directory.parent / "fit_manifest.json",
    ):
        path.unlink(missing_ok=True)


def _prediction_metadata(
    directory: Path, frame: pl.DataFrame, **extra: Any
) -> dict[str, Any]:
    path = directory / "predictions.parquet"
    return {
        "path": path.name,
        "rows": frame.height,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **extra,
    }
