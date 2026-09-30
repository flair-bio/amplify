"""Shared config schema for generic dataset-sourcing pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class DatasetSourcingConfig(BaseModel):
    """Base config for a generic download -> preprocess -> upload dataset pipeline.

    The list of known dataset names lives on the `DatasetSourceSpec` passed to
    `run_cli`/`resolve_datasets`, which validates the selection against it. This
    model only checks that a selection was made at all.
    """

    model_config = ConfigDict(extra="forbid")

    seed: int = 1957723
    datasets: list[str] | None = None
    all_datasets: bool = False
    work_dir: Path = Path("datasets/sourcing")
    preprocess: Literal["none", "minimal"] = "minimal"
    max_sequence_length: int | None = None
    create_split_subsets: bool = False
    upload_layout: Literal["per_dataset_repo", "single_repo"] = "per_dataset_repo"
    repo_owner: str | None = None
    repo_name_prefix: str = ""
    repo_id: str | None = None
    repo_prefix: str = ""
    revision: str | None = None
    commit_message: str = "Add sourced datasets"
    private: bool = True
    token: str | None = None
    force_download: bool = False
    overwrite_output: bool = False
    stats_threads: int = 8

    # Used only when an upstream dataset lacks a validation split (see
    # preprocess.create_validation_split_from_train): if MMseqs2 is
    # installed, validation rows are chosen along sequence-identity cluster
    # boundaries at this threshold instead of independently at random, so
    # validation doesn't contain near-duplicates of train sequences.
    validation_cluster_identity_threshold: float = 0.3
    validation_cluster_coverage_threshold: float = 0.8
    validation_cluster_num_threads: int = 1

    @model_validator(mode="after")
    def _validate_dataset_selection(self) -> "DatasetSourcingConfig":
        if self.all_datasets:
            return self

        if not self.datasets:
            raise ValueError(
                "No datasets selected. Set datasets in config or set all_datasets=true."
            )

        # Unknown-name validation happens later, against the source's DatasetSourceSpec.known_datasets.
        return self

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


def join_repo_path(*parts: str) -> str:
    """Join repo path fragments while removing duplicate slashes."""
    cleaned = [part.strip("/") for part in parts if part and part.strip("/")]
    return "/".join(cleaned)
