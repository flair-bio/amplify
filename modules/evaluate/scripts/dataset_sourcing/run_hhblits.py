"""Align a proteome against a sequence database with HHblits, one a3m per protein.

This is the expensive first step of calibration sourcing (minutes per protein
against BFD); run it on a cluster, then build the dataset from the alignments
with ``source_calibration.py a3m_dir=<work_dir>/a3m``. Proteins are split
round-robin across ``shard_count`` tasks (SLURM array variables by default),
and proteins whose a3m already exists are skipped, so reruns resume. Requires
HH-suite (``hhblits``) and a database such as BFD.
"""

from __future__ import annotations

import gzip
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from modules.evaluate.scripts.dataset_sourcing.shared.downloads import download_url
from modules.evaluate.scripts.dataset_sourcing.shared.pipeline import run_script

logger = logging.getLogger(__name__)

HUMAN_REFERENCE_PROTEOME_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
    "reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz"
)


class RunHHblitsConfig(BaseModel):
    """Config for aligning a proteome with HHblits."""

    model_config = ConfigDict(extra="forbid")

    # FASTA URL or local path (optionally gzipped).
    fasta: str = HUMAN_REFERENCE_PROTEOME_URL
    database: Path
    work_dir: Path = Path("eval_datasets/calibration_sourcing")
    hhblits_bin: str = "hhblits"
    # Max pairwise identity 99%, min coverage and identity with the query 50%.
    hhblits_args: list[str] = ["-id", "99", "-cov", "50", "-qid", "50"]
    cpu: int = Field(4, ge=1)
    # Default to the SLURM array task when unset.
    shard_index: int | None = Field(None, ge=0)
    shard_count: int | None = Field(None, ge=1)
    force_download: bool = False


def read_fasta(path: Path) -> dict[str, str]:
    """Return ``{id: sequence}``; UniProt ``sp|ACC|NAME`` headers map to ``ACC``."""
    opener = gzip.open if path.suffix == ".gz" else open
    records: dict[str, list[str]] = {}
    current: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:].split()[0]
                current = records.setdefault(
                    name.split("|")[1] if "|" in name else name, []
                )
            elif line:
                current.append(line)
    return {name: "".join(parts) for name, parts in records.items()}


def resolve_shard(config: RunHHblitsConfig) -> tuple[int, int]:
    task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
    task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", "0"))
    index = config.shard_index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - task_min
    count = config.shard_count or task_max - task_min + 1
    if not 0 <= index < count:
        raise ValueError(f"shard_index {index} is out of range for {count} shards.")
    return index, count


def align(config: RunHHblitsConfig, name: str, sequence: str, output: Path) -> None:
    """Run HHblits for one protein, writing ``output`` atomically."""
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        query = Path(tmp) / "query.fasta"
        query.write_text(f">{name}\n{sequence}\n")
        a3m = Path(tmp) / "out.a3m"
        subprocess.run(
            [
                config.hhblits_bin,
                "-i",
                str(query),
                "-d",
                str(config.database),
                "-oa3m",
                str(a3m),
                "-o",
                str(Path(tmp) / "out.hhr"),
                "-cpu",
                str(config.cpu),
                *config.hhblits_args,
            ],
            check=True,
            capture_output=True,
        )
        a3m.replace(output)


def run(config: RunHHblitsConfig) -> Path:
    fasta = Path(config.fasta)
    if config.fasta.startswith(("http://", "https://")):
        fasta = download_url(
            config.fasta,
            config.work_dir / "downloads" / config.fasta.rsplit("/", 1)[-1],
            force=config.force_download,
            cache_identity=config.fasta,
        )
    proteins = read_fasta(fasta)
    output_dir = config.work_dir / "a3m"
    output_dir.mkdir(parents=True, exist_ok=True)

    index, count = resolve_shard(config)
    assigned = sorted(proteins)[index::count]
    pending = [n for n in assigned if not (output_dir / f"{n}.a3m").exists()]
    logger.info(
        "Shard %d/%d: %d proteins assigned, %d already aligned.",
        index,
        count,
        len(assigned),
        len(assigned) - len(pending),
    )

    failed = []
    for i, name in enumerate(pending, 1):
        try:
            align(config, name, proteins[name], output_dir / f"{name}.a3m")
        except subprocess.CalledProcessError as exc:
            logger.error("HHblits failed for %s: %s", name, exc.stderr.decode()[-500:])
            failed.append(name)
        logger.info("Aligned %d/%d (%s).", i, len(pending), name)
    if failed:
        raise RuntimeError(f"HHblits failed for {len(failed)} proteins: {failed[:10]}")
    return output_dir


def main() -> None:
    run_script(RunHHblitsConfig, run)


if __name__ == "__main__":
    main()
