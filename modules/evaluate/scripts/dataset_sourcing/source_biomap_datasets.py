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
    "fitness_prediction",
    "metal_ion_binding",
    "localization_prediction",
    "enzyme_catalytic_efficiency",
    "optimal_ph",
    "optimal_temperature",
    "temperature_stability",
    "stability_prediction",
    "solubility_prediction",
]

BIOMAP_CARD_DESCRIPTIONS = {
    "ssp_q3": "Three-state secondary-structure prediction from protein sequence.",
    "ssp_q8": "Eight-state secondary-structure prediction from protein sequence.",
    "fold_prediction": "Protein fold classification from sequence.",
    "fitness_prediction": "Protein fitness prediction for sequence variants.",
    "metal_ion_binding": "Prediction of protein metal-ion binding labels.",
    "localization_prediction": "Prediction of protein subcellular localization.",
    "enzyme_catalytic_efficiency": "Prediction of enzyme catalytic efficiency.",
    "optimal_ph": "Prediction of a protein's optimal pH.",
    "optimal_temperature": "Prediction of a protein's optimal temperature.",
    "temperature_stability": "Prediction of protein stability across temperatures.",
    "stability_prediction": "Prediction of protein stability from sequence.",
    "solubility_prediction": "Prediction of protein solubility from sequence.",
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
