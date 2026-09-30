import os
import logging
import shlex
from pathlib import Path
from pydantic import BaseModel, ConfigDict
import subprocess
import polars as pl
import gzip
from tqdm import tqdm

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import (
    run_process,
    get_single_file,
    get_multiple_files,
)
from modules.data.src.utils.conversion_utils import batch_fasta_to_parquet

logger = logging.getLogger(__name__)

# Zero-padded decimal digits in the numeric suffix of a sequence_id
# (e.g. "{PREFIX}_000000000123"). 12 digits comfortably covers any real
# corpus (UniRef/BFD/MGnify are all <= a few billion sequences).
SEQUENCE_ID_WIDTH = 12

# Spacer used to join heavy and light chain sequences in the FASTA output.
OAS_HEAVY_LIGHT_SPACER = "XXXXX"


class PreprocessConfig(BaseModel):
    """Config. for the preprocessing step."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    test_mode: bool = True  # If True, processes only a subset of the data

    generate_fasta: bool = True
    overwrite_fasta: bool = False
    overwrite_parquet: bool = False
    convert_fasta_to_parquet: bool = True

    fasta_id_column: str = "sequence_id"
    fasta_source_id_column: str = "original_id"
    fasta_description_column: str = "description"
    fasta_sequence_column: str = "sequence"
    seqkit_split_size: int = 1_000_000  # Number of sequences per chunk

    # Extra source columns to retain in the Parquet shards (currently OAS only).
    # When set, Parquet is written directly from source, requiring convert_fasta_to_parquet=False.
    extra_columns: list[str] = []

    fasta_extract_threads: int = 8
    fasta_write_threads: int = 2
    fasta_split_threads: int = 2
    fasta_split_compression_level: int = 1  # Compression level for split FASTA
    stats_threads: int = os.cpu_count() or 1

    parquet_conversion_workers: int = 2


class PreprocessStep:
    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self._fasta_generation_registry = {
            "test_yeast": self._generate_uniref_fasta,
            "uniref": self._generate_uniref_fasta,
            "mgnify": self._generate_mgnify_fasta,
            "bfd": self._generate_bfd_fasta,
            "oas": self._generate_oas_fasta,
        }
        self.split_lines = (
            self.config.seqkit_split_size * 2
        )  # 2 lines per sequence in standard FASTA

    @staticmethod
    def _unique_id_cmd(prefix: str) -> str:
        """Returns an awk pipeline stage that rewrites each FASTA header inline.

        Replaces ``>original_header`` with ``>{PREFIX}_{n} original_header``,
        where ``n`` is the 0-indexed position of the sequence in this single
        serial stream, printed directly (no hashing/arithmetic). This is a
        bijection by construction, so it is collision-free regardless of
        dataset size.

        The original header is preserved as the description field so
        ``original_id`` can be extracted during Parquet conversion without
        any extra pass.
        """
        return (
            f"| awk 'BEGIN{{n=0}} "
            f"/^>/{{orig=substr($0,2); "
            f'printf ">%s_%0{SEQUENCE_ID_WIDTH}d %s\\n", "{prefix}", n, orig; n++; next}} '
            f"{{print}}' "
        )

    def run(self, dataset: Dataset) -> None:
        """Executes the specialized preprocessing flow for a given Dataset instance."""
        dataset.setup_directories()

        base_name = f"{dataset.name}"
        fasta_output = dataset.tmp_path / f"{base_name}_all.fasta.gz"
        split_fasta_dir = dataset.tmp_path / f"{base_name}_fasta_splits"
        parquet_output_dir = dataset.tmp_path / f"{base_name}_parquet_shards"

        split_fasta_dir.mkdir(parents=True, exist_ok=True)
        parquet_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate FASTA with unique IDs inline
        if not self.config.generate_fasta:
            logger.info("FASTA generation skipped via (generate_fasta=False).")
        else:
            if self.config.overwrite_fasta:
                logger.info(
                    f"Forcing clean regeneration of FASTA for {dataset.name}..."
                )
                fasta_output.unlink(missing_ok=True)
                for f in split_fasta_dir.glob("chunk_*.fasta.gz"):
                    f.unlink(missing_ok=True)
            else:
                logger.info(f"Generating FASTA streams for {dataset.name}...")
            self._execute_fasta_generation_fn(
                dataset, fasta_output, split_fasta_dir, parquet_output_dir
            )

        # 2. Convert split FASTAs to Parquet shards
        if not self.config.convert_fasta_to_parquet:
            logger.info(
                "Parquet conversion skipped via (convert_fasta_to_parquet=False)."
            )
        else:
            if not split_fasta_dir.exists():
                raise FileNotFoundError(
                    f"Cannot convert, FASTA splits are missing at {split_fasta_dir}."
                )
            logger.info(
                "Converting split FASTAs to partitioned Parquet dataset in parallel..."
            )
            batch_fasta_to_parquet(
                input_dir=split_fasta_dir,
                output_dir=parquet_output_dir,
                max_workers=self.config.parquet_conversion_workers,
                overwrite=self.config.overwrite_parquet,
                id_column=self.config.fasta_id_column,
                source_id_column=self.config.fasta_source_id_column,
                description_column=self.config.fasta_description_column,
                sequence_column=self.config.fasta_sequence_column,
            )

        logger.info(
            f"Preprocessing complete for {dataset.name}. Final Parquet directory at {parquet_output_dir}"
        )

    def _execute_fasta_generation_fn(
        self,
        dataset: Dataset,
        fasta_output: Path,
        split_fasta_dir: Path,
        parquet_output_dir: Path,
    ) -> None:
        """Fetches and executes the specific FASTA generation function from the registry."""
        fasta_generation_fn = self._fasta_generation_registry.get(dataset.name)
        if fasta_generation_fn is None:
            raise NotImplementedError(
                f"FASTA generation not implemented for dataset: {dataset.name}"
            )

        fasta_generation_fn(dataset, fasta_output, split_fasta_dir, parquet_output_dir)

        if not fasta_output.exists():
            raise FileNotFoundError(
                f"Expected monolithic FASTA file not found at {fasta_output}"
            )
        logger.info(
            f"Single-pass generation completed. Output: {fasta_output} and {split_fasta_dir}"
        )

    def _generate_oas_fasta(
        self,
        dataset: Dataset,
        fasta_output: Path,
        split_fasta_dir: Path,
        parquet_output_dir: Path,
    ) -> None:
        """Generates the split FASTA (for stats) and writes base Parquet shards directly.

        Unlike the other generators, OAS writes its Parquet shards itself
        (one per source CSV file) instead of going through the generic
        FASTA->Parquet converter. This lets us carry any configured
        ``extra_columns`` straight from the source CSV into the base
        Parquet, without round-tripping them through FASTA header text.
        """
        logger.info(
            "Forcing convert_fasta_to_parquet=False for OAS. Base Parquet shards are written directly during FASTA generation."
        )
        self.config.convert_fasta_to_parquet = False
        source_files = get_multiple_files(dataset.download_path, "*.csv.gz")
        if self.config.test_mode:
            source_files = list(source_files)[:2]

        bash_flags = "set -eu; " if self.config.test_mode else "set -euo pipefail; "
        prefix = dataset.name.upper()

        command = (
            f"{bash_flags}"
            f"tee >(pigz -p {self.config.fasta_write_threads} > {shlex.quote(str(fasta_output))}) "
            f"| split -l {self.split_lines} -d -a 6 "
            f"--filter='pigz -{self.config.fasta_split_compression_level} -p {self.config.fasta_split_threads} > $FILE.fasta.gz' - "
            f"{shlex.quote(str(split_fasta_dir))}/chunk_"
        )

        logger.info(
            f"Extracting heavy|light sequences from {len(source_files)} OAS CSV files..."
        )

        process = subprocess.Popen(
            ["bash", "-c", command],
            stdin=subprocess.PIPE,
            text=True,
            bufsize=8192 * 1024,  # 8 KiB > 8 MiB to reduce I/O overhead
        )
        assert process.stdin is not None

        try:
            n = 0
            for file_path in tqdm(source_files, desc="Parsing OAS CSVs"):
                df = pl.read_csv(file_path, skip_rows=1, infer_schema_length=0)

                df = df.filter(
                    ~pl.col("ANARCI_status_heavy").fill_null("").str.contains("Shorter")
                    & ~pl.col("ANARCI_status_light")
                    .fill_null("")
                    .str.contains("Shorter")
                )

                if df.is_empty():
                    continue

                file_name = file_path.name
                file_stem = file_name
                for ext in (".csv.gz", ".csv"):
                    if file_stem.endswith(ext):
                        file_stem = file_stem[: -len(ext)]
                        break

                heavy_ids = df["sequence_id_heavy"].to_list()
                light_ids = df["sequence_id_light"].to_list()
                heavy_seqs = df["sequence_alignment_aa_heavy"].to_list()
                light_seqs = df["sequence_alignment_aa_light"].to_list()

                # Precomputed so the same IDs are written to both the FASTA
                # header and the Parquet shard below.
                row_ids = [
                    f"{prefix}_{i:0{SEQUENCE_ID_WIDTH}d}" for i in range(n, n + len(df))
                ]
                # Prefixing with file_stem keeps original_id globally unique
                original_ids = [
                    f"{file_stem}:{h_id}|{l_id}"
                    for h_id, l_id in zip(heavy_ids, light_ids)
                ]
                n += len(df)

                chunk_lines = []
                for row_id, original_id, h_seq, l_seq in zip(
                    row_ids, original_ids, heavy_seqs, light_seqs
                ):
                    header = f">{row_id} {original_id}\n"
                    seq = f"{h_seq}{OAS_HEAVY_LIGHT_SPACER}{l_seq}\n"

                    chunk_lines.append(header)
                    chunk_lines.append(seq)

                process.stdin.write("".join(chunk_lines))

                # Directly build and write the base Parquet shard for this
                # source file, carrying through any configured extra_columns.
                extra_values: dict[str, pl.Series] = {}
                missing_columns = []
                for column in self.config.extra_columns:
                    if column in df.columns:
                        extra_values[column] = pl.Series(
                            column, df[column].fill_null("").to_list(), dtype=pl.String
                        )
                    else:
                        missing_columns.append(column)
                        extra_values[column] = pl.Series(
                            column, [None] * len(df), dtype=pl.String
                        )
                if missing_columns:
                    logger.warning(
                        f"{file_name}: missing configured extra_columns "
                        f"{missing_columns}; filled with null."
                    )

                frame_data = {
                    self.config.fasta_id_column: row_ids,
                    self.config.fasta_source_id_column: original_ids,
                    self.config.fasta_description_column: [
                        f"source={file_name}; locus_heavy={lh}; locus_light={ll}; "
                        f"v_call_heavy={vh}; j_call_heavy={jh}; "
                        f"v_call_light={vl}; j_call_light={jl}"
                        for lh, ll, vh, jh, vl, jl in zip(
                            df["locus_heavy"].fill_null("").to_list(),
                            df["locus_light"].fill_null("").to_list(),
                            df["v_call_heavy"].fill_null("").to_list(),
                            df["j_call_heavy"].fill_null("").to_list(),
                            df["v_call_light"].fill_null("").to_list(),
                            df["j_call_light"].fill_null("").to_list(),
                        )
                    ],
                    self.config.fasta_sequence_column: [
                        f"{h_seq}|{l_seq}"
                        for h_seq, l_seq in zip(heavy_seqs, light_seqs)
                    ],
                    **extra_values,
                }

                out_path = parquet_output_dir / f"{file_stem}.parquet"
                if out_path.exists() and not self.config.overwrite_parquet:
                    logger.debug(
                        f"Skipping Parquet write for {file_name}, already exists."
                    )
                else:
                    tmp_path = out_path.with_suffix(".parquet.tmp")
                    pl.DataFrame(frame_data).write_parquet(tmp_path)
                    tmp_path.rename(out_path)

        except Exception as e:
            process.terminate()
            raise RuntimeError(f"Failed during OAS FASTA generation: {e}") from e

        finally:
            process.stdin.close()
            process.wait()

            if process.returncode != 0:
                raise RuntimeError(
                    f"OAS pipeline failed with exit code {process.returncode}"
                )

        logger.info(f"Successfully formatted and routed {n} OAS sequences.")

    def _generate_mgnify_fasta(
        self,
        dataset: Dataset,
        fasta_output: Path,
        split_fasta_dir: Path,
        parquet_output_dir: Path,
    ) -> None:
        source_files = get_multiple_files(dataset.download_path, "*.fa.gz")
        source_files_str = " ".join(shlex.quote(str(f)) for f in source_files)
        bash_flags = "set -eu; " if self.config.test_mode else "set -euo pipefail; "
        head_cmd = f"| head -n 10000000 " if self.config.test_mode else ""

        command = (
            f"{bash_flags}"
            f"pv {source_files_str} "
            f"| pigz -dc -p {self.config.fasta_extract_threads} "
            f"| seqkit -w 0 -j {self.config.fasta_write_threads} replace -p '\\s.+' "
            f"{head_cmd}"
            + self._unique_id_cmd(dataset.name.upper())
            + f"| tee >(pigz -p {self.config.fasta_write_threads} > {shlex.quote(str(fasta_output))}) "
            f"| split -l {self.split_lines} -d -a 6 "
            f"--filter='pigz -{self.config.fasta_split_compression_level} -p {self.config.fasta_split_threads} > $FILE.fasta.gz' - "
            f"{shlex.quote(str(split_fasta_dir))}/chunk_"
        )
        run_process(["bash", "-c", command])

    def _generate_uniref_fasta(
        self,
        dataset: Dataset,
        fasta_output: Path,
        split_fasta_dir: Path,
        parquet_output_dir: Path,
    ) -> None:
        source_file = get_single_file(dataset.download_path, "*.fasta.gz")
        bash_flags = "set -eu; " if self.config.test_mode else "set -euo pipefail; "
        head_cmd = f"| head -n 10000000 " if self.config.test_mode else ""

        command = (
            f"{bash_flags}"
            f"pv -f {shlex.quote(str(source_file))} "
            f"| pigz -dc -p {self.config.fasta_extract_threads} "
            f"| seqkit seq -w 0 -j {self.config.fasta_write_threads} "
            f"{head_cmd}"
            + self._unique_id_cmd(dataset.name.upper())
            + f"| tee >(pigz -p {self.config.fasta_write_threads} > {shlex.quote(str(fasta_output))}) "
            f"| split -l {self.split_lines} -d -a 6 "
            f"--filter='pigz -{self.config.fasta_split_compression_level} -p {self.config.fasta_split_threads} > $FILE.fasta.gz' - "
            f"{shlex.quote(str(split_fasta_dir))}/chunk_"
        )
        run_process(["bash", "-c", command])

    def _generate_bfd_fasta(
        self,
        dataset: Dataset,
        fasta_output: Path,
        split_fasta_dir: Path,
        parquet_output_dir: Path,
    ) -> None:
        source_file = get_single_file(dataset.download_path, "*.tar.gz")
        bash_flags = "set -eu; " if self.config.test_mode else "set -euo pipefail; "
        test_limit = 10_000_000 if self.config.test_mode else 0

        # The BFD archive bundles multiple ffindex/ffdata pairs (a3m, cs219, hhm).
        # We only want the A3M alignments (representative + aligned sequences); the
        # cs219/hhm members are binary/profile data, not sequences, and must not be
        # streamed through the sequence-extraction awk pipeline below.
        archive_base_name = source_file.name
        if archive_base_name.endswith(".tar.gz"):
            archive_base_name = archive_base_name[: -len(".tar.gz")]
        a3m_member_name = f"{archive_base_name}_a3m.ffdata"

        # n is printed directly (no hash) — see _unique_id_cmd docstring for why.
        awk_cmd = (
            "| LC_ALL=C awk -v prefix="
            + shlex.quote(dataset.name.upper())
            + f" -v test_limit={test_limit} '"
            "BEGIN { n = 0; skip_next = 0; printed = 0 } "
            "{ "
            "line = $0; "
            r'gsub(/[^\t -~]/, "", line); '
            r"if (line ~ /^#/) next; "
            "if (skip_next) { skip_next = 0; next } "
            r"if (line ~ /consensus/) { skip_next = 1; next } "
            r"is_header = (line ~ /^>/); "
            r"if (is_header && (line ~ /^>sp/ || line ~ /^>tr/ || line ~ /^>UP/)) { skip_next = 1; next } "
            "if (is_header) { "
            r'split(line, parts, " "); '
            "id = substr(parts[1], 2); "
            f'printf ">%s_%0{SEQUENCE_ID_WIDTH}d %s\\n", prefix, n, id; '
            "n++; "
            "} else { "
            "line = toupper(line); "
            r'gsub(/-/, "", line); '
            "print line; "
            "} "
            "if (test_limit > 0) { printed++; if (printed >= test_limit) exit } "
            "}' "
        )

        command = (
            f"{bash_flags}"
            rf"pv -f {shlex.quote(str(source_file))} "
            rf"| pigz -dc -p {self.config.fasta_extract_threads} "
            rf"| tar -Oxf - {shlex.quote(a3m_member_name)} "
            + awk_cmd
            + f"| tee >(pigz -p {self.config.fasta_write_threads} > {shlex.quote(str(fasta_output))}) "
            f"| split -l {self.split_lines} -d -a 6 "
            f"--filter='pigz -{self.config.fasta_split_compression_level} -p {self.config.fasta_split_threads} > $FILE.fasta.gz' - "
            f"{shlex.quote(str(split_fasta_dir))}/chunk_"
        )
        run_process(["bash", "-c", command])
