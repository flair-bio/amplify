# Utilities for input/output operations, such as downloading files and running shell commands, that are used multiple places.
import logging
import selectors
import shutil
from collections.abc import Sequence
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO, cast
import polars as pl


logger = logging.getLogger(__name__)


def resolve_path(
    override: str | Path | None, default: Path, label: str, *, required: bool = False
) -> Path | None:
    """Resolve a path from an optional override string, falling back to a default.

    Returns the resolved path if it exists, or None if it does not.

    Args:
        override: An explicit path string that takes priority over ``default``.
        default:  Fallback path used when ``override`` is None.
        label:    Human-readable name for the path (used in log/error messages).
        required: If True, raise FileNotFoundError instead of returning None.

    Raises:
        FileNotFoundError: If ``required=True`` and the resolved path does not exist.
    """
    path = Path(override) if override else default
    if path.exists():
        return path
    if required:
        raise FileNotFoundError(f"{label} path not found: {path}")
    logger.warning("%s path not found at %s — will be skipped.", label, path)
    return None


def override_or_default(override: str | Path | None, default: Path) -> Path:
    """Return ``Path(override)`` if provided, otherwise ``default``.

    Used for output paths (which need not already exist) where an explicit
    override should take priority over a computed default location.
    """
    return Path(override) if override else default


def resolve_parquet_source(source: str | Path) -> pl.LazyFrame:
    """Return a LazyFrame for a source that is either a single file or a directory of shards.

    Accepts a path to a single .parquet file or a path to a dir. containing one or more
    .parquet files (shards). All shards are scanned and unioned in sorted filename order.
    """
    path = Path(source)
    if path.is_dir():
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"No .parquet files found in directory: {path}")
        logger.debug("Scanning %d shard(s) from directory: %s", len(shards), path)
        return pl.scan_parquet([str(s) for s in shards])
    return pl.scan_parquet(str(path))


def run_process(command_list: Sequence[str], cwd: str | Path | None = None) -> None:
    """Run a command without a shell and stream stdout/stderr as they arrive.

    The command must be passed as a tokenized sequence instead of a shell string.
    That keeps argument handling explicit, avoids shell-injection hazards, and
    makes the helper suitable for library code rather than only top-level CLI
    scripts.

    Output is streamed in real time so long-running commands remain observable.
    Standard output and standard error naturally inherit the parent's file
    descriptors, ensuring the highest fidelity interactive terminal behavior
    (e.g., preserving progress bars) while bypassing Python-level loggers.

    Args:
        command_list: Executable plus arguments, already split into tokens.
        cwd: Optional working directory for the child process.

    Raises:
        ValueError: If ``command_list`` is empty.
        NotADirectoryError: If ``cwd`` is provided but does not exist as a directory.
        FileNotFoundError: If the executable in ``command_list[0]`` cannot be found.
        subprocess.CalledProcessError: If the child exits with a non-zero status.
    """
    if not command_list:
        raise ValueError("command_list cannot be empty.")

    working_directory = Path(cwd) if cwd is not None else None
    if working_directory is not None and not working_directory.is_dir():
        raise NotADirectoryError(
            f"Working directory does not exist: {working_directory}"
        )

    try:
        # Omitting stdout and stderr lets the subprocess natively inherit fd 1 and 2.
        with subprocess.Popen(
            list(command_list),
            cwd=working_directory,
        ) as process:
            return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command_list))

    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find the executable '{command_list[0]}'. Is it installed and in your PATH?"
        ) from exc


def download_with_aria2c(
    urls: list[str],
    download_dir: str,
    is_multiple: bool,
    split: int = 16,
    max_connections: int = 16,
    min_split_size: str = "1M",
    max_concurrent: int = 5,
) -> None:
    """Handles the actual aria2c subprocess execution."""

    if not shutil.which("aria2c"):
        raise FileNotFoundError("aria2c is not installed on this system.")

    command = ["aria2c", "-c", "--dir", str(download_dir)]

    if is_multiple:
        # Configuration for downloading MANY separate files at once
        command.extend(["-Z", "-j", str(max_concurrent)])
    else:
        # Configuration for downloading ONE massive file very fast
        command.extend(
            ["-s", str(split), "-x", str(max_connections), "-k", min_split_size]
        )

    # Append all URLs at the end
    command.extend(urls)
    run_process(command)


def get_single_file(directory: Path, pattern: str) -> Path:
    """Finds exactly one file matching the pattern in a directory."""
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly 1 file matching '{pattern}' in {directory}, found {len(files)}."
        )
    return files[0]


def get_multiple_files(directory: Path, pattern: str) -> list[Path]:
    """Finds one or more files matching the pattern in a directory."""
    files = list(directory.glob(pattern))
    if not files:
        raise ValueError(
            f"Expected at least 1 file matching '{pattern}' in {directory}, found none."
        )
    return files
