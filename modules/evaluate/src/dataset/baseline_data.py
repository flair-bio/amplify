"""Baseline observations, identities, loading, validation and row alignment."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import polars as pl
from datasets import Dataset, DatasetDict

from modules.evaluate.src.contracts.config import (
    BaselineName,
    BaselinesConfig,
    DatasetWorkspaceConfig,
    dataset_source_identity,
)

BASELINE_PROTOCOLS = {
    "majority_class": "train-majority-v1",
    "train_mean": "train-mean-v1",
    "composition": "aa-composition-logistic-or-ridge-v1",
    "mmseqs_knn": "mmseqs-easy-search-top1-v1",
    "amino_acid_frequency": "test-corpus-iid-aa-frequency-v1",
}
SEPARATION_PRIOR_PROTOCOL = "casp-original-sequence-separation-v1"


def identity_hash(payload: Any) -> str:
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(payload):
        digest.update(chunk.encode())
    return digest.hexdigest()


def sequence_rows(
    dataset: Dataset,
    *,
    sequence_column: str,
    label_column: str,
    id_column: str,
    task_type: str,
    num_labels: int | None,
) -> list[dict[str, Any]]:
    """Preserve source observations, including repeated IDs."""
    if not len(dataset):
        raise ValueError("A shared baseline requires nonempty train and test data.")
    rows = []
    columns = [sequence_column, label_column]
    if id_column in dataset.column_names:
        columns.append(id_column)
    for index, row in enumerate(dataset.select_columns(columns)):
        label = row[label_column]
        if task_type == "token_classification":
            if not isinstance(label, list) or len(label) != len(row[sequence_column]):
                raise ValueError("Token baseline requires one label per residue.")
            if any(
                not isinstance(value, int)
                or (value != -100 and not 0 <= value < (num_labels or 0))
                for value in label
            ):
                raise ValueError("Token baseline class indices are outside num_labels.")
        elif not isinstance(label, (int, float)) or not math.isfinite(label):
            raise ValueError("Sequence baseline targets must be finite scalars.")
        elif task_type == "sequence_classification":
            if label != int(label) or not 0 <= label < (num_labels or 0):
                raise ValueError("Baseline class indices are outside num_labels.")
            label = int(label)
        else:
            label = float(label)
        identifier = row.get(id_column)
        rows.append(
            {
                "id": index if identifier is None else identifier,
                "sequence": row[sequence_column],
                "label": label,
            }
        )
    if task_type == "token_classification":
        ids = [json.dumps(row["id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Shared token baselines require unique source protein IDs."
            )
    return rows


def unlabeled_sequence_rows(
    dataset: Dataset, *, sequence_column: str, id_column: str
) -> list[dict[str, Any]]:
    """Preserve label-free sequence observations, including repeated source IDs."""
    if not len(dataset):
        raise ValueError("A shared baseline requires nonempty test data.")
    columns = [sequence_column]
    if id_column in dataset.column_names:
        columns.append(id_column)
    rows = []
    for index, row in enumerate(dataset.select_columns(columns)):
        sequence = row[sequence_column]
        if not isinstance(sequence, str) or not sequence:
            raise ValueError(
                "Amino-acid-frequency baseline requires nonempty sequences."
            )
        identifier = row.get(id_column)
        rows.append(
            {"id": index if identifier is None else identifier, "sequence": sequence}
        )
    return rows


def baseline_context(
    workspace: DatasetWorkspaceConfig,
    dataset: DatasetDict,
    names: list[BaselineName],
    *,
    sequence_column: str,
    label_column: str,
    id_column: str,
    split_roles: dict[str, Any],
    max_length: int | None = None,
) -> dict[str, Any]:
    """Describe expected artifacts before model-specific preparation."""
    BaselinesConfig(names=names).validate_task(workspace.task_type)
    dataset_identity = dataset_source_identity(
        workspace.dataset_repo_id or workspace.dataset_name,
        workspace.dataset_config_name,
        workspace.dataset_revision,
    )
    if workspace.task_type == "pseudo_perplexity":
        rows = unlabeled_sequence_rows(
            dataset["test"],
            sequence_column=sequence_column,
            id_column=id_column,
        )
        fit = {
            "version": 2,
            "dataset": dataset_identity,
            "task_type": workspace.task_type,
            "fit_policy": "test_corpus_unsupervised",
            "protocols": BASELINE_PROTOCOLS,
            "columns": [sequence_column, id_column],
            "max_residues": None if max_length is None else max_length - 2,
            "corpus_fingerprint": identity_hash(rows),
        }
        return {
            "fit": fit,
            "test": {
                "dataset": dataset_identity,
                "split": split_roles["test"],
                "fingerprint": identity_hash(rows),
            },
            "root": str((workspace.dataset_root / "baselines").resolve()),
            "names": names,
        }
    arguments: dict[str, Any] = {
        "sequence_column": sequence_column,
        "label_column": label_column,
        "id_column": id_column,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
    }
    train_rows = sequence_rows(dataset["train"], **arguments)
    test_rows = sequence_rows(dataset["test"], **arguments)
    fit = {
        "version": 2,
        "dataset": dataset_identity,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
        "fit_policy": "train_only",
        "protocols": BASELINE_PROTOCOLS,
        "split_roles": {"name": split_roles["name"], "train": split_roles["train"]},
        "columns": [sequence_column, label_column, id_column],
        "train_fingerprint": identity_hash(train_rows),
    }
    if workspace.task_type == "token_classification":
        fit["fit_policy"] = "all_labeled_training_residues"
        fit["protocols"] = {"majority_class": "all-train-valid-residue-majority-v1"}
    return {
        "fit": fit,
        "test": {
            "dataset": fit["dataset"],
            "split": split_roles["test"],
            "fingerprint": identity_hash(test_rows),
        },
        "root": str((workspace.dataset_root / "baselines").resolve()),
        "names": names,
    }


def contact_baseline_context(
    workspace: DatasetWorkspaceConfig,
    dataset: Any,
    names: list[BaselineName],
) -> dict[str, Any]:
    """Describe model-independent CASP contact artifacts."""
    BaselinesConfig(names=names).validate_task(workspace.task_type)
    if workspace.task_type != "contact_prediction" or names != ["separation_prior"]:
        raise ValueError("Contact baseline context requires separation_prior only.")
    source = dataset_source_identity(
        workspace.dataset_repo_id or workspace.dataset_name,
        workspace.dataset_config_name,
        workspace.dataset_revision,
    )
    identity = {
        "source": source,
        "revision": dataset.revision,
        "fingerprint": dataset.fingerprint,
        "schema_version": dataset.schema_version,
    }
    return {
        "fit": {
            "version": 1,
            "dataset": identity,
            "task_type": workspace.task_type,
            "fit_policy": "model_independent",
            "protocols": {"separation_prior": SEPARATION_PRIOR_PROTOCOL},
        },
        "test": {
            "dataset": identity,
            "split": "test",
        },
        "root": str((workspace.dataset_root / "baselines").resolve()),
        "names": names,
    }


def external_baseline_context(
    context: dict[str, Any],
    workspace: DatasetWorkspaceConfig,
    dataset: DatasetDict,
    *,
    dataset_id: str,
    split: str,
    config_name: str | None = None,
    revision: str | None = None,
    sequence_column: str,
    label_column: str,
    id_column: str,
    max_length: int | None = None,
) -> dict[str, Any]:
    if workspace.task_type == "pseudo_perplexity":
        rows = unlabeled_sequence_rows(
            dataset["test"],
            sequence_column=sequence_column,
            id_column=id_column,
        )
        external_identity = dataset_source_identity(dataset_id, config_name, revision)
        fit = {
            **context["fit"],
            "dataset": external_identity,
            "max_residues": None if max_length is None else max_length - 2,
            "corpus_fingerprint": identity_hash(rows),
        }
        return {
            **context,
            "fit": fit,
            "test": {
                "dataset": external_identity,
                "split": split,
                "fingerprint": identity_hash(rows),
            },
        }
    rows = sequence_rows(
        dataset["test"],
        sequence_column=sequence_column,
        label_column=label_column,
        id_column=id_column,
        task_type=workspace.task_type,
        num_labels=workspace.num_labels,
    )
    return {
        **context,
        "test": {
            "dataset": dataset_source_identity(dataset_id, config_name, revision),
            "split": split,
            "fingerprint": identity_hash(rows),
        },
    }


def artifact_identity(context: dict[str, Any], name: str) -> dict[str, Any]:
    fit = context["fit"]
    if name == "mmseqs_knn":
        fit = {
            **fit,
            "protocols": {
                **fit["protocols"],
                "mmseqs_knn": "mmseqs-significant-top1-training-prior-v2",
            },
        }
    return {"baseline": name, "fit": fit, "test": context["test"]}


def mmseqs_coverage(frame: pl.DataFrame) -> dict[str, int | float]:
    """Summarize significant-hit coverage without dropping fallback rows."""
    fallback = frame["fallback"]
    if fallback.dtype != pl.Boolean or fallback.null_count() or not frame.height:
        raise ValueError("MMseqs2 baseline requires nonnull boolean fallback flags.")
    fallback_count = int(fallback.sum())
    matched = frame.height - fallback_count
    return {
        "matched": matched,
        "fallback": fallback_count,
        "total": frame.height,
        "fraction_matched": matched / frame.height,
    }


def artifact_directory(context: dict[str, Any], name: str) -> Path:
    identity = artifact_identity(context, name)
    return (
        Path(context["root"])
        / name
        / f"fit-{identity_hash(identity['fit'])[:12]}"
        / f"test-{identity_hash(identity['test'])[:12]}"
    )


def load_baseline_artifact(
    context: dict[str, Any], name: str
) -> tuple[pl.DataFrame, Path]:
    directory = artifact_directory(context, name)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared baseline '{name}' at {directory}. "
            "Run run_evaluate.py --baselines with the same dataset/task "
            "configuration before scoring."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != artifact_identity(context, name):
        raise ValueError(f"Incompatible baseline manifest: {manifest_path}")
    path = directory / "predictions.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"Corrupt baseline predictions: {path}")
    frame = pl.read_parquet(path)
    if name == "separation_prior":
        required = {
            "id",
            "record_id",
            "unit_id",
            "split",
            "source",
            "edition",
            "complex_id",
            "target",
            "kind",
            "labels_available",
            "original_i",
            "original_j",
            "labels",
            "scores",
            "length",
            "status",
        }
    else:
        required = {"id", "prediction"}
    if name not in {"amino_acid_frequency", "separation_prior"}:
        required.add("label")
    if context["fit"]["task_type"] in {
        "sequence_classification",
        "token_classification",
    }:
        required.add("probability")
    if name == "mmseqs_knn":
        required.add("fallback")
    if not required.issubset(frame.columns) or frame.height != manifest["rows"]:
        raise ValueError(f"Invalid baseline prediction schema/count: {path}")
    if name == "mmseqs_knn" and manifest.get("coverage") != mmseqs_coverage(frame):
        raise ValueError(f"Invalid MMseqs2 coverage metadata: {manifest_path}")
    return frame, path


def align_token_baseline(
    baseline: pl.DataFrame, predictions: pl.DataFrame
) -> pl.DataFrame:
    """Broadcast a stored prior onto live or cached evaluated residue windows."""
    source = {json.dumps(row["id"]): row for row in baseline.to_dicts()}
    if len(source) != baseline.height:
        raise ValueError("Shared token baseline contains duplicate source IDs.")
    observed: dict[str, list[int]] = defaultdict(list)
    rows = []
    for row in predictions.select("id", "label").to_dicts():
        key = json.dumps(row["id"])
        if key not in source:
            raise ValueError(
                "Shared token baseline IDs do not match model observations."
            )
        prior = source[key]
        labels = row["label"]
        ragged = isinstance(labels, list)
        labels = labels if ragged else [labels]
        observed[key].extend(labels)
        rows.append(
            {
                "id": row["id"],
                "prediction": (
                    [prior["prediction"]] * len(labels)
                    if ragged
                    else prior["prediction"]
                ),
                "probability": (
                    [prior["probability"]] * len(labels)
                    if ragged
                    else prior["probability"]
                ),
            }
        )
    for key, labels in observed.items():
        original = [label for label in source[key]["label"] if label != -100]
        # Cropping and ignored residues leave a contiguous window of valid labels.
        if not any(
            original[start : start + len(labels)] == labels
            for start in range(len(original) - len(labels) + 1)
        ):
            raise ValueError(
                "Shared token baseline labels do not match an evaluated residue window."
            )
    frame = pl.DataFrame(rows)
    return frame if "probability" in predictions.columns else frame.drop("probability")


def align_baseline(
    baseline: pl.DataFrame,
    predictions: pl.DataFrame,
    *,
    regression: bool,
    match_labels: bool = True,
) -> pl.DataFrame:
    """Align repeated observations without multiplying rows in an ID join."""
    if not match_labels:

        def id_key(identifier: Any) -> str:
            return json.dumps(identifier, allow_nan=False)

        id_positions: dict[str, int] = {}
        for index, identifier in enumerate(baseline["id"].to_list()):
            key = id_key(identifier)
            if key in id_positions:
                raise ValueError("Label-free baseline IDs must be unique.")
            id_positions[key] = index
        indices = []
        seen = set()
        for identifier in predictions["id"].to_list():
            key = id_key(identifier)
            if key not in id_positions or key in seen:
                raise ValueError(
                    "Label-free baseline IDs do not match model observations."
                )
            seen.add(key)
            indices.append(id_positions[key])
        if len(indices) != len(id_positions):
            raise ValueError(
                "Shared baseline and model have different observation counts."
            )
        return baseline[indices]

    def row_key(identifier: Any, label: Any) -> str:
        # Prediction parquet and torch regression labels use float32.
        if regression:
            label = struct.unpack("f", struct.pack("f", float(label)))[0]
        return json.dumps([identifier, label], allow_nan=False)

    positions: dict[str, deque[int]] = defaultdict(deque)
    for index, row in enumerate(baseline.select("id", "label").iter_rows()):
        positions[row_key(*row)].append(index)
    indices = []
    for row in predictions.select("id", "label").iter_rows():
        candidates = positions[row_key(*row)]
        if not candidates:
            raise ValueError(
                "Shared baseline IDs/labels do not match model observations."
            )
        indices.append(candidates.popleft())
    if any(positions.values()):
        raise ValueError("Shared baseline and model have different observation counts.")
    return baseline[indices].drop("label")
