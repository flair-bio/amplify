"""Manifest loading and residue-validated embeddings for latent alignment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from modules.post_train.src.config import DatasetConfig

ROW_COLUMNS = (
    "record_id",
    "protein_id",
    "chain",
    "source",
    "split",
    "cluster_id",
    "residue_index",
    "residue_id",
    "amino_acid",
)


@dataclass(frozen=True)
class EmbeddingSpec:
    feature_columns: tuple[str, ...]
    teacher: dict[str, Any]

    @property
    def dimension(self) -> int:
        return len(self.feature_columns)


def embedding_spec(metadata: dict[str, Any]) -> EmbeddingSpec:
    if not isinstance(metadata, dict):
        raise ValueError("Manifest metadata must be an object.")
    if metadata.get("embedding_format") != "parquet_per_protein_v1":
        raise ValueError("Require embedding_format=parquet_per_protein_v1.")
    schema = metadata.get("embedding_schema")
    if not isinstance(schema, dict):
        raise ValueError("Manifest must declare embedding feature columns.")
    columns = schema.get("feature_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
        or set(columns) & set(ROW_COLUMNS)
    ):
        raise ValueError(
            "Manifest feature columns must be nonempty, unique names distinct from row metadata."
        )
    teacher = metadata.get("structural_teacher")
    if teacher is None:
        teacher = {}
    if not isinstance(teacher, dict):
        raise ValueError("structural_teacher must be a metadata object when supplied.")
    for declaration in (teacher, schema):
        if "dimension" in declaration and (
            type(declaration["dimension"]) is not int
            or declaration["dimension"] != len(columns)
        ):
            raise ValueError(
                "Declared embedding dimension differs from manifest feature columns."
            )
    return EmbeddingSpec(tuple(columns), teacher)


class Record(BaseModel):
    """Preserve producer provenance fields while validating consumed fields."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str = Field(min_length=1)
    protein_id: str = Field(min_length=1)
    chain: str
    source: str = Field(min_length=1)
    split: str
    sequence: str
    residue_ids: list[str]
    cluster_id: str | None = None
    embedding_file: Path
    embedding_spec: EmbeddingSpec = Field(exclude=True)


@dataclass(frozen=True)
class PreparedDataset:
    train: list[Record]
    validation: list[Record]
    metadata: dict[str, Any]
    embedding_spec: EmbeddingSpec
    manifest: Path
    hub_revision: str | None = None


def load_dataset(config: DatasetConfig) -> PreparedDataset:
    root = Path(config.path).expanduser()
    revision = None
    if root.is_dir():
        if config.revision is not None:
            raise ValueError("dataset.revision applies only to Hub datasets.")
        root = root.absolute()
    elif root.exists():
        raise NotADirectoryError(
            f"dataset.path must point to a dataset directory, not a file: {root}"
        )
    elif root.is_absolute() or config.path.startswith((".", "~")):
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    else:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=config.path,
                repo_type="dataset",
                revision=config.revision,
            )
        )
        revision = root.name
    return load_manifest(root / config.manifest_file, hub_revision=revision)


def load_manifest(
    path: Path | str, *, hub_revision: str | None = None
) -> PreparedDataset:
    # Do not resolve the final symlink: Hub manifests point into a blob cache,
    # but their relative embedding paths belong to the snapshot directory.
    path = Path(path).expanduser().absolute()
    payload = json.loads(path.read_text())
    metadata = payload["metadata"]
    spec = embedding_spec(metadata)
    records = [
        Record.model_validate({**row, "embedding_spec": spec})
        for row in payload["records"]
    ]
    if len({row.id for row in records}) != len(records):
        raise ValueError("Duplicate protein IDs in manifest.")
    resolved = []
    for row in records:
        if not re.fullmatch(r"[A-Z]+", row.sequence):
            raise ValueError(
                f"{row.id}: expected a nonempty uppercase residue sequence."
            )
        if len(row.residue_ids) != len(row.sequence) or len(
            set(row.residue_ids)
        ) != len(row.sequence):
            raise ValueError(f"{row.id}: invalid residue identities/order.")
        if row.split not in {"train", "validation"}:
            raise ValueError(f"{row.id}: split must be train or validation.")
        relative = row.embedding_file
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{row.id}: embedding_file must stay within the dataset.")
        # HF snapshot files are symlinks into its blob cache; lexical confinement
        # is checked above, rather than rejecting those legitimate symlinks.
        target = path.parent / relative
        if target.suffix != ".parquet" or not target.is_file():
            raise ValueError(
                f"{row.id}: missing or incompatible .parquet target: {target}"
            )
        resolved.append(row.model_copy(update={"embedding_file": target}))
    train = sorted(
        (row for row in resolved if row.split == "train"), key=lambda row: row.id
    )
    validation = [row for row in resolved if row.split == "validation"]
    sequences = {row.sequence for row in train}
    if len(sequences) != len(train):
        raise ValueError("Training sequences must be distinct.")
    if not train or not validation or sequences & {row.sequence for row in validation}:
        raise ValueError(
            "Require nonempty, sequence-disjoint train and validation splits."
        )
    return PreparedDataset(
        train=train,
        validation=validation,
        metadata=metadata,
        embedding_spec=spec,
        manifest=path,
        hub_revision=hub_revision,
    )


def load_embeddings(row: Record) -> NDArray[np.float32]:
    table = pq.read_table(row.embedding_file)
    spec = row.embedding_spec
    names = [*ROW_COLUMNS, *spec.feature_columns]
    if table.column_names != names or table.num_rows != len(row.sequence):
        raise ValueError(f"{row.id}: incompatible Parquet columns or row count.")
    if (
        table.schema.field("residue_index").type != pa.int32()
        or any(
            table.schema.field(name).type != pa.float32()
            for name in spec.feature_columns
        )
        or any(table[name].null_count for name in names if name != "cluster_id")
    ):
        raise ValueError(f"{row.id}: invalid Parquet types or null values.")
    expected_metadata = {
        b"embedding_format": b"residue-by-dimension-float32",
        b"embedding_dimension": str(spec.dimension).encode(),
    }
    metadata = table.schema.metadata or {}
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"{row.id}: incompatible Parquet embedding metadata.")
    for field, key in (
        ("model_id", b"teacher_model"),
        ("revision", b"teacher_revision"),
    ):
        expected = spec.teacher.get(field)
        if (
            expected is not None
            and key in metadata
            and metadata[key] != str(expected).encode()
        ):
            raise ValueError(
                f"{row.id}: Parquet teacher metadata differs from manifest."
            )
    for column, value in {
        "record_id": row.id,
        "protein_id": row.protein_id,
        "chain": row.chain,
        "source": row.source,
        "split": row.split,
        "cluster_id": row.cluster_id,
    }.items():
        if table[column].to_pylist() != [value] * len(row.sequence):
            raise ValueError(f"{row.id}: Parquet {column} differs from manifest.")
    for column, expected in {
        "residue_index": list(range(len(row.sequence))),
        "residue_id": row.residue_ids,
        "amino_acid": list(row.sequence),
    }.items():
        if table[column].to_pylist() != expected:
            raise ValueError(f"{row.id}: Parquet {column} is not residue-aligned.")
    values = np.column_stack([table[name].to_numpy() for name in spec.feature_columns])
    if (
        values.dtype != np.float32
        or values.shape != (len(row.sequence), spec.dimension)
        or not np.isfinite(values).all()
        or not np.all(np.any(values != 0, axis=1))
    ):
        raise ValueError(
            f"{row.id}: expected finite, nonzero float32[L,{spec.dimension}] targets."
        )
    return values
