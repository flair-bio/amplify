"""Standalone sequence baselines and validated prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Literal

import polars as pl
from datasets import Dataset, DatasetDict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from modules.evaluate.src.artifacts.identity import dataset_identity
from modules.evaluate.src.dataset.workspace import DatasetWorkspaceConfig
from modules.evaluate.src.predictors import composition, mmseqs_knn, priors

BaselineName = Literal[
    "majority_class",
    "train_mean",
    "composition",
    "mmseqs_knn",
    "separation_prior",
    "amino_acid_frequency",
]

#: Explicit routing: which baselines each task accepts.
TASK_BASELINES: dict[str, frozenset[str]] = {
    "sequence_classification": frozenset(
        {"majority_class", "composition", "mmseqs_knn"}
    ),
    "sequence_regression": frozenset({"train_mean", "composition", "mmseqs_knn"}),
    "token_classification": frozenset({"majority_class"}),
    "contact_prediction": frozenset({"separation_prior"}),
    "pseudo_perplexity": frozenset({"amino_acid_frequency"}),
}

#: Tasks that accept shared baselines at all.
SUPPORTED_BASELINE_TASKS = frozenset(TASK_BASELINES)

#: Tasks whose baselines are fitted by the structural runner (not the sequence
#: runner). Their names are supplied by ``--baselines`` rather than the task
#: YAML, because model runs of these tasks reject ``baselines.names``.
STRUCTURAL_TASK_BASELINES: dict[str, tuple[str, ...]] = {
    "contact_prediction": ("separation_prior",),
    "pseudo_perplexity": ("amino_acid_frequency",),
}

#: Baselines fitted from a declared reference corpus rather than the task's
#: own training split. The reference must be pinned to a full commit SHA.
REFERENCE_BASELINES = frozenset({"amino_acid_frequency"})

#: Fixed, versioned protocol per baseline. Each protocol enters the artifact
#: identity, so changing a setting creates a new artifact directory.
BASELINE_PROTOCOLS: dict[str, dict[str, Any]] = {
    "majority_class": {
        "version": 1,
        "rule": "most frequent training class; ties to the lowest class index",
    },
    "train_mean": {"version": 1, "rule": "arithmetic mean of training targets"},
    "composition": composition.COMPOSITION_PROTOCOL,
    "mmseqs_knn": mmseqs_knn.MMSEQS_KNN_PROTOCOL,
    "separation_prior": priors.SEPARATION_PRIOR_PROTOCOL,
    "amino_acid_frequency": priors.AMINO_ACID_FREQUENCY_PROTOCOL,
}

#: Baselines whose artifacts carry per-class probabilities.
CLASSIFICATION_BASELINES = frozenset({"majority_class", "composition", "mmseqs_knn"})

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class AminoAcidReference(BaseModel):
    """Declared public reference corpus for the amino-acid frequency prior.

    Only the training split at an immutable 40-hex commit SHA is accepted, so
    the pseudo-perplexity test split can never supply the frequencies.
    """

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str
    split: Literal["train"] = "train"
    sequence_column: str = "sequence"

    @field_validator("revision")
    @classmethod
    def _require_commit_sha(cls, value: str) -> str:
        if not _HEX40.fullmatch(value):
            raise ValueError(
                "amino_acid_reference.revision must be a full 40-hex commit SHA"
            )
        return value


class BaselinesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[BaselineName] = Field(default_factory=list)
    amino_acid_reference: AminoAcidReference | None = None

    def validate_task(self, task_type: str) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError("baselines.names must not contain duplicates.")
        if not self.names:
            return
        if task_type not in TASK_BASELINES:
            raise ValueError(
                f"Shared baselines are not supported for task_type='{task_type}'; "
                "select a task with an implemented shared baseline."
            )
        for name in self.names:
            if name not in TASK_BASELINES[task_type]:
                owners = sorted(
                    t for t, names in TASK_BASELINES.items() if name in names
                )
                requirement = " or ".join(f"'{t}'" for t in owners)
                raise ValueError(
                    f"Shared baseline '{name}' requires task_type={requirement} "
                    f"(got '{task_type}')."
                )
        if set(self.names) & REFERENCE_BASELINES and self.amino_acid_reference is None:
            raise ValueError(
                "amino_acid_frequency requires baselines.amino_acid_reference "
                "(repo_id plus a 40-hex commit SHA of a train-only corpus)."
            )
        if self.amino_acid_reference is not None and not (
            set(self.names) & REFERENCE_BASELINES
        ):
            raise ValueError(
                "baselines.amino_acid_reference is only used by amino_acid_frequency."
            )


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
            if not isinstance(label, (list, tuple)) or len(label) != len(
                row[sequence_column]
            ):
                raise ValueError(
                    "Token baseline targets require one label per residue."
                )
            if any(
                not isinstance(value, int)
                or (value != -100 and not 0 <= value < (num_labels or 0))
                for value in label
            ):
                raise ValueError("Baseline class indices are outside num_labels.")
            positions = [i for i, value in enumerate(label) if value != -100]
            rows.append(
                {
                    "id": row.get(id_column, index),
                    "sequence": row[sequence_column],
                    "label": [label[i] for i in positions],
                    "evaluated_positions": positions,
                }
            )
            continue
        if not isinstance(label, (int, float)) or not math.isfinite(label):
            raise ValueError("Sequence baseline targets must be finite scalars.")
        if task_type == "sequence_classification":
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
) -> dict[str, Any]:
    """Describe expected artifacts before model-specific preparation."""
    BaselinesConfig(names=names).validate_task(workspace.task_type)
    arguments: dict[str, Any] = {
        "sequence_column": sequence_column,
        "label_column": label_column,
        "id_column": id_column,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
    }
    train_rows = sequence_rows(dataset["train"], **arguments)
    test_rows = sequence_rows(dataset["test"], **arguments)
    source_identity = _workspace_dataset_identity(workspace)
    fit = {
        "version": 1,
        "dataset": source_identity,
        "task_type": workspace.task_type,
        "num_labels": workspace.num_labels,
        "fit_policy": "train_only",
        "split_roles": {"name": split_roles["name"], "train": split_roles["train"]},
        "columns": [sequence_column, label_column, id_column],
        "train_fingerprint": identity_hash(train_rows),
    }
    return {
        "fit": fit,
        "test": {
            "dataset": source_identity,
            "split": split_roles["test"],
            "fingerprint": identity_hash(test_rows),
        },
        "root": str((workspace.dataset_root / "baselines").resolve()),
        "names": names,
    }


def _workspace_dataset_identity(workspace: DatasetWorkspaceConfig) -> dict[str, Any]:
    return dataset_identity(
        source=workspace.dataset_repo_id or workspace.dataset_name,
        subset=workspace.dataset_subset,
        revision=workspace.dataset_revision,
    )


def external_baseline_context(
    context: dict[str, Any],
    workspace: DatasetWorkspaceConfig,
    dataset: DatasetDict,
    *,
    dataset_id: str,
    split: str,
    sequence_column: str,
    label_column: str,
    id_column: str,
    dataset_revision: str | None = None,
    dataset_subset: str | None = None,
) -> dict[str, Any]:
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
            "dataset": dataset_identity(
                source=dataset_id, subset=dataset_subset, revision=dataset_revision
            ),
            "split": split,
            "fingerprint": identity_hash(rows),
        },
    }


def artifact_identity(context: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "baseline": name,
        "protocol": BASELINE_PROTOCOLS[name],
        "fit": context["fit"],
        "test": context["test"],
    }


def artifact_directory(context: dict[str, Any], name: str) -> Path:
    identity = artifact_identity(context, name)
    return (
        Path(context["root"])
        / name
        / f"protocol-{identity_hash(identity['protocol'])[:12]}"
        / f"fit-{identity_hash(identity['fit'])[:12]}"
        / f"test-{identity_hash(identity['test'])[:12]}"
    )


def build_sequence_baseline(
    name: BaselineName,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    num_labels: int | None,
) -> pl.DataFrame:
    """Fit on train rows only and predict every test row.

    ``num_labels`` selects the task: an integer means classification and
    ``None`` means regression.
    """
    labels = [row["label"] for row in train_rows]
    test_labels = [row["label"] for row in test_rows]
    test_ids = [row["id"] for row in test_rows]
    probabilities: list[list[float]] | None = None
    prediction: list[Any]
    if name == "majority_class":
        token_task = "evaluated_positions" in train_rows[0]
        if token_task:
            labels = [value for row in labels for value in row]
        if not labels:
            raise ValueError("Majority baseline requires valid training labels.")
        frequencies = Counter(labels)
        counts = [frequencies[index] for index in range(num_labels or 0)]
        winner = max(range(len(counts)), key=counts.__getitem__)
        prediction = (
            [[winner] * len(row) for row in test_labels]
            if token_task
            else [winner] * len(test_rows)
        )
        if not token_task:
            probabilities = [[count / len(labels) for count in counts]] * len(test_rows)
    elif name == "train_mean":
        prediction = [math.fsum(labels) / len(labels)] * len(test_rows)
    else:
        sequences = [row["sequence"] for row in train_rows]
        queries = [row["sequence"] for row in test_rows]
        if name == "composition":
            if num_labels is None:
                model = composition.fit_composition_regressor(sequences, labels)
                prediction = composition.predict_composition_regressor(
                    model, queries
                ).tolist()
            else:
                model = composition.fit_composition_classifier(
                    sequences, [int(label) for label in labels], num_labels=num_labels
                )
                predicted, probs = composition.predict_composition_classifier(
                    model, queries, num_labels=num_labels
                )
                prediction = predicted.tolist()
                probabilities = probs.tolist()
        else:
            hits = mmseqs_knn.run_search(queries, sequences)
            if num_labels is None:
                prediction = mmseqs_knn.knn_regress(
                    hits, train_labels=labels, num_queries=len(queries)
                ).tolist()
            else:
                predicted, probs = mmseqs_knn.knn_classify(
                    hits,
                    train_labels=[int(label) for label in labels],
                    num_queries=len(queries),
                    num_labels=num_labels,
                )
                prediction = predicted.tolist()
                probabilities = probs.tolist()
    data: dict[str, Any] = {
        "id": test_ids,
        "label": test_labels,
        "prediction": prediction,
    }
    if probabilities is not None:
        data["probability"] = probabilities
    if "evaluated_positions" in test_rows[0]:
        data["evaluated_positions"] = [row["evaluated_positions"] for row in test_rows]
    return pl.DataFrame(data)


def load_baseline_artifact(
    context: dict[str, Any], name: str
) -> tuple[pl.DataFrame, Path]:
    directory = artifact_directory(context, name)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shared baseline '{name}' at {directory}. "
            "Run run_evaluate.py --baselines with the same dataset/task "
            "configuration, then rerun the comparison (no inference is needed)."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("identity") != artifact_identity(context, name):
        raise ValueError(f"Incompatible baseline manifest: {manifest_path}")
    path = directory / "predictions.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256"):
        raise ValueError(f"Corrupt baseline predictions: {path}")
    frame = pl.read_parquet(path)
    required = {"id", "label", "prediction"}
    if (
        name in CLASSIFICATION_BASELINES
        and context["fit"]["num_labels"] is not None
        and context["fit"]["task_type"] != "token_classification"
    ):
        required.add("probability")
    if context["fit"]["task_type"] == "token_classification":
        required.add("evaluated_positions")
    if not required.issubset(frame.columns) or frame.height != manifest["rows"]:
        raise ValueError(f"Invalid baseline prediction schema/count: {path}")
    return frame, path


def align_baseline(
    baseline: pl.DataFrame, predictions: pl.DataFrame, *, regression: bool
) -> pl.DataFrame:
    """Align repeated observations without multiplying rows in an ID join."""
    if "evaluated_positions" in baseline.columns:
        return _align_token_baseline(baseline, predictions)

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


def _align_token_baseline(
    baseline: pl.DataFrame, predictions: pl.DataFrame
) -> pl.DataFrame:
    if "evaluated_positions" not in predictions.columns:
        raise ValueError(
            "Token comparison requires evaluated_positions in saved predictions; "
            "recover the original crop/mask coordinates before comparing. "
            "Labels alone cannot identify residue positions."
        )
    observations: dict[str, dict[int, tuple[int, int]]] = {}
    for row in baseline.iter_rows(named=True):
        key = json.dumps(row["id"])
        if key in observations:
            raise ValueError("Token comparison requires unique source example IDs.")
        positions, labels, values = (
            row["evaluated_positions"],
            row["label"],
            row["prediction"],
        )
        if len(positions) != len(set(positions)) or not (
            len(positions) == len(labels) == len(values)
        ):
            raise ValueError("Invalid baseline residue positions/masks.")
        observations[key] = dict(zip(positions, zip(labels, values)))
    seen: set[tuple[str, int]] = set()
    aligned = []
    for row in predictions.iter_rows(named=True):
        key = json.dumps(row["id"])
        positions, labels = row["evaluated_positions"], row["label"]
        if len(positions) != len(labels) or len(labels) != len(row["prediction"]):
            raise ValueError("Token prediction positions/masks have different lengths.")
        values = []
        for position, label in zip(positions, labels):
            observation = observations.get(key, {}).get(position)
            if observation is None or observation[0] != label:
                raise ValueError(
                    "Shared baseline IDs/residue positions/masks/labels do not match."
                )
            if (key, position) in seen:
                raise ValueError("Repeated evaluated residue in model predictions.")
            seen.add((key, position))
            values.append(observation[1])
        aligned.append(values)
    return pl.DataFrame(
        {"id": predictions["id"], "prediction": aligned},
        schema={"id": predictions.schema["id"], "prediction": pl.List(pl.Int64)},
    )
