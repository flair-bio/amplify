"""One-off script to source datasets from the biomap-research organization on Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path

from modules.evaluate.src.dataset_sourcing.config import DatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.pipeline import run_cli
from modules.evaluate.src.dataset_sourcing.spec import DatasetSourceSpec

BIOMAP_DATASETS: list[str] = [
    "ssp_q3",
    "ssp_q8",
    "fold_prediction",
    "localization_prediction",
    "stability_prediction",
    "contact_prediction_binary",
]

BIOMAP_CARD_DESCRIPTIONS = {
    "ssp_q3": "Three-state secondary-structure prediction from protein sequence.",
    "ssp_q8": "Eight-state secondary-structure prediction from protein sequence.",
    "fold_prediction": "Protein fold classification from sequence.",
    "localization_prediction": "Prediction of protein subcellular localization.",
    "stability_prediction": "Prediction of protein stability from sequence.",
    "contact_prediction_binary": "Sparse residue-pair contacts for protein contact-map prediction.",
}


class SourceBiomapDatasetsConfig(DatasetSourcingConfig):
    """Config for sourcing biomap-research datasets."""

    work_dir: Path = Path("datasets/biomap_research_sourcing")
    repo_name_prefix: str = "biomap-research-"
    repo_prefix: str = "biomap"
    commit_message: str = "Add sourced biomap-research datasets"


# Normalize source column names to the shared evaluation dataset schema.
BIOMAP_SOURCE_SPEC = DatasetSourceSpec(
    known_datasets=BIOMAP_DATASETS,
    repo_id_template="biomap-research/{dataset_name}",
    column_rename={"seq": "sequence", "label": "targets"},
    card_descriptions=BIOMAP_CARD_DESCRIPTIONS,
)


def main() -> None:
    """CLI entrypoint for sourcing one or more biomap-research datasets."""
    run_cli(sys.argv[1:], model_cls=SourceBiomapDatasetsConfig, spec=BIOMAP_SOURCE_SPEC)


if __name__ == "__main__":
    main()
