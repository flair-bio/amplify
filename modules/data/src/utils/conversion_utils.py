"""utils.conversion_utils
Utilities to convert massive FASTA datasets to Parquet
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import polars_bio as pb
import subprocess

logger = logging.getLogger(__name__)


def _convert_single_fasta(
    input_file: Path,
    output_file: Path,
    id_column: str,
    source_id_column: str,
    description_column: str,
    sequence_column: str,
    compression: str,
    compression_level: int,
) -> str:
    """Worker function to process a single FASTA file with atomic writes."""
    try:
        lf = pb.scan_fasta(str(input_file))
        schema_names = set(lf.collect_schema().names())

        if "sequence" not in schema_names:
            raise ValueError(f"Missing required column: sequence")

        if "name" in schema_names:
            base_name = pl.col("name").fill_null("")
            base_description = (
                pl.col("description").fill_null("")
                if "description" in schema_names
                else pl.lit("")
            )
        elif "description" in schema_names:
            header = pl.col("description").fill_null("")
            base_name = header.str.extract(r"^(\S+)\s*(.*)?$", group_index=1).fill_null(
                ""
            )
            base_description = header.str.extract(
                r"^(\S+)\s*(.*)?$", group_index=2
            ).fill_null("")
        else:
            raise ValueError("Missing required columns: name/description")

        parsed_lf = (
            lf.with_columns(
                base_name.alias("_normalized_name"),
                base_description.alias("_normalized_description"),
            )
            .with_columns(
                # id: the unique header ID injected by _unique_id_cmd (first field)
                pl.col("_normalized_name").alias(id_column),
                # source_id: first word of description — the original pre-unique-ID header
                pl.col("_normalized_description")
                .str.extract(r"^(\S+)", group_index=1)
                .fill_null("")
                .alias(source_id_column),
                # description: annotation after the original_id (first word stripped)
                pl.col("_normalized_description")
                .str.replace(r"^\S+\s*", "")
                .alias(description_column),
                pl.col("sequence").alias(sequence_column),
            )
            .select(id_column, source_id_column, description_column, sequence_column)
        )

        temp_output = output_file.with_suffix(".parquet.tmp")
        parsed_lf.sink_parquet(
            str(temp_output),
            compression=compression,  # type: ignore
            compression_level=compression_level,
            row_group_size=100_000,
        )
        temp_output.rename(output_file)

        return str(input_file)

    except Exception as exc:
        raise RuntimeError(f"Failed processing {input_file.name}: {exc}") from exc


def batch_fasta_to_parquet(
    input_dir: str | Path,
    output_dir: str | Path,
    max_workers: int = 4,
    overwrite: bool = False,
    compression: str = "zstd",
    compression_level: int = 3,
    id_column: str = "id",
    source_id_column: str = "source_id",
    description_column: str = "description",
    sequence_column: str = "sequence",
) -> None:
    """Converts a directory of split FASTA files into a Parquet dataset.

    Scans `input_dir` for FASTA files, checks `output_dir` to see if they've already been completely
    processed, and uses a ProcessPoolExecutor to convert the remaining files in parallel.

    Args:
        input_dir: Directory containing pre-split FASTA files.
        output_dir: Destination directory for Parquet files.
        max_workers: Number of concurrent files to process. (Recommend 2-4 for large gzip files).
        compression: Parquet compression codec.
        compression_level: Compression strength.
    """
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()

    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Directory does not exist: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all standard fasta extensions in the directory
    valid_extensions = {".fasta", ".fa", ".fasta.gz", ".fa.gz"}
    input_files = [
        p
        for p in input_path.iterdir()
        if p.is_file() and any(p.name.lower().endswith(ext) for ext in valid_extensions)
    ]

    if not input_files:
        logger.warning(f"No FASTA files found in {input_path}")
        return

    tasks = []
    for in_file in input_files:
        # Strip extension correctly regardless of .fa or .fasta.gz
        base_name = in_file.name
        for ext in valid_extensions:
            if base_name.lower().endswith(ext):
                base_name = base_name[: -len(ext)]
                break

        out_file = output_path / f"{base_name}.parquet"

        # Core Pipeline Resumability: Skip if the parquet file already exists
        if out_file.exists() and not overwrite:
            logger.debug(f"Skipping {in_file.name}, Parquet already exists.")
            continue

        # Clean up any leftover temp files from a previous crashed run
        tmp_file = out_file.with_suffix(".parquet.tmp")
        if tmp_file.exists():
            tmp_file.unlink()

        tasks.append((in_file, out_file))

    if not tasks:
        logger.info("All files are already processed. Dataset is fully up to date.")
        return

    logger.info(f"Found {len(tasks)} pending files out of {len(input_files)} total.")
    logger.info(f"Starting ProcessPoolExecutor with {max_workers} workers...")

    # ProcessPoolExecutor bypasses the GIL and safely isolates memory per process
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _convert_single_fasta,
                in_file,
                out_file,
                id_column,
                source_id_column,
                description_column,
                sequence_column,
                compression,
                compression_level,
            ): in_file
            for in_file, out_file in tasks
        }

        for future in as_completed(futures):
            in_file = futures[future]
            try:
                result_path = future.result()
                success_count += 1
                logger.info(
                    f"[{success_count}/{len(tasks)}] Processed: {Path(result_path).name}"
                )
            except Exception as exc:
                logger.error(f"Job failed for {in_file.name}: {exc}")

    logger.info(
        f"Batch conversion complete. Successfully processed {success_count}/{len(tasks)} files."
    )
