import csv
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.data.src.config import DataPipelineConfig
from modules.data.src.dataset.dataset import Dataset, DatasetConfig
from modules.data.src.steps.preprocess import PreprocessConfig, PreprocessStep


def test_oas_nano_config_and_individual_steps(tmp_path):
    configs = Path(__file__).parents[2] / "modules/data/configs"
    for step in ("download", "preprocess", "score"):
        config = load_and_parse(
            [configs / "config.yaml", configs / "oas_nano.yaml"],
            DataPipelineConfig,
            overrides=[
                f"steps.{step}.enabled=true",
                f"dataset.base_path={tmp_path}",
            ],
        )
        assert [
            name
            for name in type(config.steps).model_fields
            if getattr(config.steps, name).enabled
        ] == [step]
        dataset = Dataset(config.dataset)
        assert dataset.download_path == tmp_path / "oas_nano/download"
        assert dataset.tmp_path == tmp_path / "oas_nano/tmp"


def test_oas_nano_preprocess_preserves_single_chains_and_annotations(tmp_path):
    dataset = Dataset(DatasetConfig(name="oas_nano", base_path=str(tmp_path)))
    dataset.setup_directories()
    # Create files in reverse order to exercise deterministic IDs.
    for file_name in ("B.csv.gz", "A.csv.gz"):
        with gzip.open(dataset.download_path / file_name, "wt", newline="") as f:
            f.write(json.dumps({"Species": "alpaca", "Isotype": "Bulk"}) + "\n")
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence_id",
                    "sequence_aa",
                    "sequence_alignment_aa",
                    "ANARCII_status",
                ],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "sequence_id": "same_id",
                        "sequence_aa": sequence,
                        "sequence_alignment_aa": sequence[:-3],
                        "ANARCII_status": "Shorter than IMGT defined: fw1",
                    }
                    for sequence in ("EVQLVESGGGLVQPGGSLRLSCAAS", "QVQLVQSGAEVKKPGASVKVSCKAS")
                ]
                + [{"sequence_id": "empty", "sequence_aa": ""}]
            )

    step = PreprocessStep(
        PreprocessConfig(
            test_mode=False,
            seqkit_split_size=1,
            extra_columns=["ANARCII_status", "Redundancy"],
        )
    )
    step.run(dataset)

    shards = sorted((dataset.tmp_path / "oas_nano_parquet_shards").glob("*.parquet"))
    frame = pl.read_parquet(shards)
    assert frame["sequence_id"].to_list() == [
        f"OAS_NANO_{i:012d}" for i in range(4)
    ]
    assert frame["original_id"].to_list() == [
        "A:same_id", "A:same_id", "B:same_id", "B:same_id"
    ]
    assert frame["sequence"].to_list() == [
        "EVQLVESGGGLVQPGGSLRLSCAAS", "QVQLVQSGAEVKKPGASVKVSCKAS"
    ] * 2
    assert frame["ANARCII_status"].to_list() == [
        "Shorter than IMGT defined: fw1"
    ] * 4
    assert frame["Redundancy"].null_count() == 4
    assert "species=alpaca; isotype=Bulk" in frame["description"][0]
    with gzip.open(dataset.tmp_path / "oas_nano_all.fasta.gz", "rt") as f:
        fasta = f.read()
    split_text = ""
    for path in sorted((dataset.tmp_path / "oas_nano_fasta_splits").glob("*.gz")):
        with gzip.open(path, "rt") as f:
            split_text += f.read()
    assert fasta == split_text
    assert fasta.splitlines()[1::2] == frame["sequence"].to_list()
    assert step.config.convert_fasta_to_parquet is False


def test_oas_nano_preprocess_rejects_missing_amino_acid_column(tmp_path):
    dataset = Dataset(DatasetConfig(name="oas_nano", base_path=str(tmp_path)))
    dataset.setup_directories()
    with gzip.open(dataset.download_path / "bad.csv.gz", "wt") as f:
        f.write('{}\nsequence_id,sequence\nid,ACGT\n')
    with pytest.raises(ValueError, match="missing required columns.*sequence_aa"):
        PreprocessStep(PreprocessConfig()).run(dataset)
