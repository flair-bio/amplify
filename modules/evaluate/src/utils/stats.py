import logging
import gzip
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

COLS_TO_KEEP = [
    "file",
    "type",
    "num_seqs",
    "sum_len",
    "min_len",
    "avg_len",
    "max_len",
    "Q1",
    "Q2",
    "Q3",
    "sum_gap",
    "sum_n",
]


def _run_seqkit(
    command: Sequence[str],
    output_path: Path,
    input_files: Sequence[Path] | None = None,
) -> None:
    """Run SeqKit without relying on platform-specific shell utilities."""
    try:
        with output_path.open("wb") as output_handle:
            if input_files is None:
                subprocess.run(list(command), stdout=output_handle, check=True)
                return

            process = subprocess.Popen(
                list(command), stdin=subprocess.PIPE, stdout=output_handle
            )
            write_error: BrokenPipeError | None = None
            try:
                assert process.stdin is not None
                for input_file in input_files:
                    opener = gzip.open if input_file.suffix == ".gz" else open
                    with opener(input_file, "rb") as input_handle:
                        shutil.copyfileobj(input_handle, process.stdin)
            except BrokenPipeError as exc:
                write_error = exc
            finally:
                if process.stdin is not None:
                    process.stdin.close()

            return_code = process.wait()
            if write_error is not None:
                raise subprocess.CalledProcessError(
                    return_code or 1, list(command)
                ) from write_error
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, list(command))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not find 'seqkit'. Install SeqKit and ensure it is on PATH."
        ) from exc


def _run_stats_and_read(
    command: Sequence[str],
    tmp_tsv: Path,
    input_files: Sequence[Path] | None = None,
) -> pl.DataFrame:
    """Run a SeqKit command and read the resulting TSV."""
    _run_seqkit(command, tmp_tsv, input_files=input_files)
    df = pl.read_csv(tmp_tsv, separator="\t")

    available_cols = [c for c in COLS_TO_KEEP if c in df.columns]
    return df.select(available_cols)


def compute_seqkit_stats(
    split_fasta_dir: Path,
    fasta_output: Path,
    stats_path: Path,
    dataset_name: str,
    threads: int = 8,
) -> None:
    """Runs multi-threaded seqkit stats on FASTA shards and saves to a single Parquet file."""
    fasta_files = sorted(list(split_fasta_dir.glob("*.fasta.gz")))
    if not fasta_files:
        fasta_files = sorted(list(split_fasta_dir.glob("*.fasta")))

    if not fasta_files:
        logger.warning(
            f"No FASTA targets detected in {split_fasta_dir} to extract metrics."
        )
        return

    # Determine final Parquet output path
    if stats_path.is_dir():
        output_parquet_path = stats_path / f"{dataset_name}_stats.parquet"
    else:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        output_parquet_path = stats_path

    with tempfile.TemporaryDirectory(dir=split_fasta_dir) as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_stats_tsv = tmp_dir / "stats.tsv"

        # --- Case 1: Single file path ---
        if len(fasta_files) == 1:
            command = [
                "seqkit",
                "stats",
                "-b",
                "-a",
                "-T",
                "-j",
                str(threads),
                str(fasta_files[0]),
            ]
            df = _run_stats_and_read(command, tmp_stats_tsv)
            df.write_parquet(output_parquet_path)
            logger.info(
                f"Successfully saved single-sharded statistics to {output_parquet_path}"
            )
            return

        # --- Case 2: Multi-file path ---
        tmp_overall_tsv = tmp_dir / "overall.tsv"

        if fasta_output.exists():
            # Build an explicit combined list (shards + pre-existing overall file)
            command = [
                "seqkit",
                "stats",
                "-b",
                "-a",
                "-T",
                "-j",
                str(threads),
                *(str(path) for path in [*fasta_files, fasta_output]),
            ]
            df = _run_stats_and_read(command, tmp_stats_tsv)

            # Check against .name because '-b' outputs basenames, not full paths
            df = df.with_columns(
                pl.when(pl.col("file") == fasta_output.name)
                .then(pl.lit("overall"))
                .otherwise(pl.col("file"))
                .alias("file")
            )
        else:
            # Generate shard statistics
            shard_command = [
                "seqkit",
                "stats",
                "-b",
                "-a",
                "-T",
                "-j",
                str(threads),
                *(str(path) for path in fasta_files),
            ]
            df_shards = _run_stats_and_read(shard_command, tmp_stats_tsv)

            # Generate total summary statistics by streaming all shards through
            # Python, avoiding GNU xargs, cat, and the optional pigz utility.
            overall_command = [
                "seqkit",
                "stats",
                "-a",
                "-T",
                "-j",
                str(threads),
                "-",
            ]
            df_overall = _run_stats_and_read(
                overall_command, tmp_overall_tsv, input_files=fasta_files
            )
            df_overall = df_overall.with_columns(pl.lit("overall").alias("file"))

            df = pl.concat([df_shards, df_overall])

        df.write_parquet(output_parquet_path)
        logger.info(f"Successfully compiled metrics to {output_parquet_path}")
