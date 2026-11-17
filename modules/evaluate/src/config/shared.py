"""Shared configuration components and source identities."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


_MIN_RECOMMENDED_BOOTSTRAP_SAMPLES = 1000


_DEFAULT_N_PERMUTATIONS = 1000


BASELINE_NAMES = frozenset(
    {
        "majority_class",
        "train_mean",
        "composition",
        "mmseqs_knn",
        "amino_acid_frequency",
    }
)


DEFAULT_SPLIT_NAME = "default"


CLASSICAL_HEAD_TYPES = frozenset({"sklearn_linear", "knn", "random_forest", "xgboost"})


TORCH_EMBEDDING_HEAD_TYPES = frozenset({"torch_linear"})


BaselineName = Literal[
    "majority_class",
    "train_mean",
    "composition",
    "mmseqs_knn",
    "amino_acid_frequency",
    "separation_prior",
    "human_residue",
    "blosum62",
]


ControlCondition = Literal[
    "untrained",
    "random_trunk",
    "random_head",
    "random_both",
    "scrambled_labels",
    "scrambled_sequences",
]


HeadType = Literal[
    "bilinear512",
    "attention_l1",
    "categorical_jacobian",
    "mlp",
    "torch_linear",
    "sklearn_linear",
    "knn",
    "random_forest",
    "xgboost",
]


EmbeddingPooling = Literal["mean", "cls"]


EmbeddingDtype = Literal["float16", "float32"]


EmbeddingAutocastDtype = Literal["float16", "bfloat16", "float32"]


TestSubsets = dict[
    str, dict[str, StrictStr | StrictInt | StrictFloat | StrictBool | None]
]


def validate_metric_group_column(value: str | None) -> str | None:
    if value is not None and (
        not value.strip() or value in {"id", "prediction", "label", "probability"}
    ):
        raise ValueError("group_column must name a non-reserved source column.")
    return value


def validate_test_subsets(value: TestSubsets | None) -> TestSubsets:
    if value is None:
        return {}
    for name, filters in value.items():
        validate_directory_label(name)
        if not filters or any(not column.strip() for column in filters):
            raise ValueError("Test subsets require nonempty column-equality filters.")
        if any(
            isinstance(item, float) and not math.isfinite(item)
            for item in filters.values()
        ):
            raise ValueError("Test subset filter values must be finite.")
    return value


def dataset_source_identity(
    dataset_id: str, config_name: str | None = None, revision: str | None = None
) -> str:
    """Distinguish configurations/revisions sharing the same repo or suite root."""
    if config_name is None and revision is None:
        return dataset_id
    return json.dumps([dataset_id, config_name, revision])


class DatasetWorkspaceConfig(BaseModel):
    """Dataset/task settings shared by model and baseline runs."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str  # dataset name that serves as a mnemonic for the dataset being evaluated, e.g. "stability_prediction"
    dataset_repo_id: str | None = (
        None  # path to HuggingFace dataset repo e.g. biomap-research/stability_prediction
    )
    dataset_config_name: str | None = Field(
        None, min_length=1, description="Named configuration within a dataset suite."
    )
    dataset_revision: str | None = None
    task_type: str
    default_task_type: str | None = None  # dataset default evaluation method
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
        from modules.evaluate.src.tasks import available_tasks

        if value not in available_tasks():
            raise ValueError(
                f"Unknown task_type '{value}'. Available: {available_tasks()}."
            )
        return value

    @field_validator("default_task_type")
    @classmethod
    def _validate_default_task_type(cls, value: str | None) -> str | None:
        if value is not None:
            cls._validate_task_type(value)
        return value

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
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    tmp_path: str | None = Field(
        None,
        description=(
            "Temporary cache root, optionally on HPC scratch storage. Defaults to "
            "<benchmark_root>/tmp; task/dataset/model subdirectories are appended."
        ),
    )

    @field_validator("run_id", "model_name")
    @classmethod
    def _validate_directory_label(cls, value: str) -> str:
        return validate_directory_label(value)

    @field_validator("model_name")
    @classmethod
    def _validate_model_label(cls, value: str) -> str:
        if value == "baselines":
            raise ValueError("model_name='baselines' is reserved for shared baselines.")
        return value


def validate_directory_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "Output labels must start with a letter or digit and contain "
            "only letters, digits, '.', '_', or '-'."
        )
    return value


class EmbeddingCacheConfig(BaseModel):
    """Frozen-trunk feature extraction and cache settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    pooling: EmbeddingPooling = Field(
        "mean",
        description=(
            "How to pool (B, S, H) trunk hidden states into (B, H). Must "
            "use 'mean' for residue-only masked mean pooling, or 'cls' "
            "for first-token pooling."
        ),
    )
    layers: list[int] | None = Field(
        default=None,
        description=(
            "Indices into the model's returned hidden-state tuple. Negative "
            "indices follow Python indexing; multiple layers are pooled "
            "independently then concatenated. If omitted while the cache is "
            "enabled, [-1] selects the final layer."
        ),
    )
    dtype: EmbeddingDtype = Field(
        "float32", description="Precision embeddings are stored/computed in."
    )
    autocast_dtype: EmbeddingAutocastDtype = Field(
        "float32",
        description=(
            "Precision used inside the frozen-trunk forward pass. float32 "
            "disables autocast; float16/bfloat16 autocast is opt-in."
        ),
    )
    normalize_before_pooling: bool = Field(
        True,
        description=(
            "L2-normalize each token representation before mean pooling. "
            "Disable explicitly to retain raw representation magnitudes."
        ),
    )
    max_cache_gb: float = Field(
        2.0,
        description=(
            "Refuse to write a split's embeddings to disk if they would "
            "exceed this many GB; falls back to an in-memory-only cache for "
            "that run instead of risking filling shared storage."
        ),
    )
    cache_policy: Literal["reuse", "refresh", "off"] = Field(
        "reuse",
        description=(
            "'reuse': load a matching cached split from disk if present. "
            "'refresh': always recompute and overwrite. 'off': compute "
            "in-memory only, never read or write the disk cache."
        ),
    )

    @field_validator("layers")
    @classmethod
    def _validate_layers(cls, layers: list[int] | None) -> list[int] | None:
        if layers is None:
            return None
        if not layers:
            raise ValueError(
                "embedding_cache.layers must contain at least one layer index."
            )
        if len(set(layers)) != len(layers):
            raise ValueError(
                "embedding_cache.layers must not contain duplicate indices."
            )
        return layers

    @property
    def resolved_layers(self) -> list[int]:
        """Return the effective layers for an enabled embedding cache."""
        return self.layers if self.layers is not None else [-1]


class WandbConfig(BaseModel):
    """Config. for optional Weights & Biases logging."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    project: str = "flair-probe"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = Field(default_factory=list)
