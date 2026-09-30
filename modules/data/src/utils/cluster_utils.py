"""Shared helpers for clustering pipeline path handling and numeric normalization."""

from pathlib import Path
import shutil


def validate_num_threads(num_threads: int) -> None:
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}.")


def normalize_threshold(name: str, value: float) -> float:
    """Normalize threshold values to [0, 1], accepting fraction (0.9) or percent (90)."""
    normalized = value / 100.0 if value > 1.0 else value
    if not 0.0 < normalized <= 1.0:
        raise ValueError(f"{name} must be in the range (0, 1], got {value}.")
    return normalized


def format_threshold_label(
    threshold: float,
    decimal_separator: str,
) -> str:
    # Emit percentage-style labels so 0.9 -> 90 and 0.6 -> 60.
    normalized = normalize_threshold("threshold", threshold)
    return str(int(round(normalized * 100)))


def write_success_marker(directory: Path) -> None:
    (directory / "_SUCCESS").touch()


def is_complete_dir(directory: Path) -> bool:
    return directory.is_dir() and (directory / "_SUCCESS").is_file()


def promote_tmp_dir(tmp_dir: Path, final_dir: Path) -> None:
    """Atomically publish a completed tmp workspace as the final directory."""
    write_success_marker(tmp_dir)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.replace(final_dir)


def stage_tmp_dir(final_dir: Path, *, clean: bool = True) -> Path:
    """Return sibling ``.tmp`` staging dir for ``final_dir``, ready for writes."""
    tmp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if clean and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir
