"""Shared config schema for generic dataset-sourcing pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MIN_SOURCE_LABEL_COUNT = 5
DEFAULT_MIN_TRAIN_LABEL_SUPPORT = 20
DEFAULT_MIN_HOLDOUT_LABEL_SUPPORT = 3


class HubUploadConfig(BaseModel):
    """Hugging Face Hub upload target shared by every sourcing script."""

    model_config = ConfigDict(extra="forbid")

    # per_dataset_repo: <repo_owner>/<repo_name_prefix><dataset>; single_repo: repo_id/<repo_prefix>/<dataset>.
    upload_layout: Literal["per_dataset_repo", "single_repo"] = "per_dataset_repo"
    repo_owner: str | None = None
    repo_name_prefix: str = ""
    repo_id: str | None = None
    repo_prefix: str = ""
    # Hub branch/ref to commit to; unrelated to DatasetSourcingConfig.source_revision (the input pin).
    revision: str | None = None
    commit_message: str = "Add sourced datasets"
    # Applies when the repo is created; shipped YAMLs override it to false.
    private: bool = True
    # None falls back to the HF_TOKEN / HUGGINGFACE_HUB_TOKEN env vars.
    token: str | None = None

    def resolve_upload_repo_id(self, dataset_name: str) -> str:
        """Resolve the target dataset repo id for the selected upload layout."""
        if self.upload_layout == "single_repo":
            if not self.repo_id:
                raise ValueError(
                    "repo_id is required when upload_layout is 'single_repo'"
                )
            return self.repo_id

        owner = self.repo_owner
        if not owner and self.repo_id:
            # Fall back to single_repo's owner so per_dataset_repo can reuse the same repo_id config.
            owner = self.repo_id.split("/", 1)[0]

        if not owner:
            raise ValueError(
                "repo_owner is required when upload_layout is 'per_dataset_repo'"
            )

        repo_name = f"{self.repo_name_prefix}{dataset_name}".strip("/")
        return f"{owner}/{repo_name}"

    def has_upload_target(self) -> bool:
        """Return True when this config contains enough info to perform uploads."""
        if self.upload_layout == "single_repo":
            return bool(self.repo_id)
        return bool(self.repo_owner or self.repo_id)


class DatasetSourcingConfig(HubUploadConfig):
    """Base config for a generic download -> preprocess -> upload dataset pipeline.

    The list of known dataset names lives on the `DatasetSourceSpec` passed to
    `run_cli`/`resolve_datasets`, which validates the selection against it.
    """

    # Seeds every random draw and cluster tie-break, so reruns reproduce the same splits.
    seed: int = 1957723
    datasets: list[str] | None = None
    all_datasets: bool = False
    work_dir: Path = Path("datasets/sourcing")
    preprocess: Literal["none", "minimal"] = "minimal"
    # Longer sequences are dropped, not truncated; None keeps everything.
    max_sequence_length: int | None = None
    create_split_subsets: bool = False
    # Upstream HF commit to download; None tracks the default branch (not reproducible).
    source_revision: str | None = None
    force_download: bool = False
    # Never reuse existing outputs; source scripts that call prepare_output_dir also delete them first.
    overwrite_output: bool = False
    # Threads for SeqKit stats only; MMseqs2 uses cluster_num_threads.
    stats_threads: int = 8

    # MMseqs2 settings for every cluster-based split: synthesized validation,
    # full-dataset splits, and hold-cluster-out subsets.
    cluster_identity_threshold: float = 0.3
    cluster_coverage_threshold: float = 0.8
    cluster_num_threads: int = 1


UniProtReviewStatus = Literal["reviewed", "unreviewed", "all"]


class UniProtDatasetSourcingConfig(DatasetSourcingConfig):
    """Shared organism and review filters for UniProt-backed sources."""

    # NCBI taxonomy id (human: 9606); None means all organisms.
    organism_id: int | None = Field(default=None, ge=1)
    review_status: UniProtReviewStatus = "all"
    # Target row shares for cluster-based canonical splits; must sum to 1.
    split_fractions: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )


class UniProtLabelSupportConfig(UniProtDatasetSourcingConfig):
    """Minimum label-support thresholds for multi-label UniProt sources."""

    min_label_count: int = Field(
        DEFAULT_MIN_SOURCE_LABEL_COUNT,
        ge=1,
        description=(
            "Minimum source-protein support before splitting; final label "
            "vocabulary is also filtered by canonical split support."
        ),
    )
    min_train_label_count: int = Field(DEFAULT_MIN_TRAIN_LABEL_SUPPORT, ge=1)
    min_validation_label_count: int = Field(DEFAULT_MIN_HOLDOUT_LABEL_SUPPORT, ge=1)
    min_test_label_count: int = Field(DEFAULT_MIN_HOLDOUT_LABEL_SUPPORT, ge=1)


def join_repo_path(*parts: str) -> str:
    """Join repo path fragments while removing duplicate slashes."""
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)
