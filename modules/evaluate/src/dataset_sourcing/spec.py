"""Source-specific glue plugged into the generic dataset-sourcing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSourceSpec:
    """Describes the parts of the pipeline that differ per data source."""

    known_datasets: list[str]
    repo_id_template: str  # e.g. "biomap-research/{dataset_name}"
    column_rename: dict[str, str] = field(default_factory=dict)
    card_source_line: str = (
        "Sourced from `{repo_id}` and prepared for Hugging Face datasets usage."
    )

    def source_repo_id(self, dataset_name: str) -> str:
        """Resolve the upstream HF repo id to download a dataset from."""
        return self.repo_id_template.format(dataset_name=dataset_name)
