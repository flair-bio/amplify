import ctypes
import logging
import mmap
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from mmap import MADV_RANDOM as MADV_RANDOM  # re-exported for callers
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class FileId(NamedTuple):
    """A file's identity, unlike its path, which symlinks and hard links alias."""

    device: int
    inode: int


@dataclass
class MappingResult:
    stat: os.stat_result | None = None
    num_bytes: int = 0
    errno: int | None = None  # set if madvise or stat failed


@dataclass(frozen=True)
class AdviseResults:
    paths: dict[Path, MappingResult]
    num_bytes: int  # bytes of mappings advised, each file counted once
    num_files: int  # distinct files stat'd: aliased paths count once


@dataclass(frozen=True)
class MemoryMapping:
    """One mapping (line) of /proc/<pid>/maps.

    https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html
    """

    start: int
    end: int
    device: int
    inode: int
    path: Path | None

    @classmethod
    def from_line(cls, line: str) -> "MemoryMapping":
        try:
            address, _perms, _offset, dev, inode, *path = line.rstrip("\n").split(
                maxsplit=5
            )
            start, end = (int(x, 16) for x in address.split("-"))
            major, minor = (int(x, 16) for x in dev.split(":"))
            return cls(
                start=start,
                end=end,
                device=os.makedev(major, minor),
                inode=int(inode),
                path=Path(path[0]) if path else None,
            )
        except ValueError as e:
            raise ValueError(f"not a /proc/<pid>/maps line: {line!r}") from e

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"invalid address range {self.start:#x}-{self.end:#x}")
        if self.start % mmap.PAGESIZE or self.end % mmap.PAGESIZE:
            raise ValueError(
                f"address range {self.start:#x}-{self.end:#x} is not page-aligned"
            )

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def id(self) -> FileId:
        return FileId(self.device, self.inode)


def read_memory_mappings(
    proc_maps: Path = Path("/proc/self/maps"),
) -> list[MemoryMapping]:
    """Parse every mapping in proc_maps, skipping lines that don't parse.

    Raises OSError if proc_maps can't be read.
    """
    mappings = []
    # Read the entire file at once since it can change mid-read.
    # See: https://docs.kernel.org/filesystems/proc.html#process-specific-subdirectories
    # Also, no splitlines() since a path may contain other line breaks.
    for line in proc_maps.read_text().split("\n"):
        if not line:
            continue
        try:
            mappings.append(MemoryMapping.from_line(line))
        except ValueError as e:
            logger.debug(f"Skipping memory-map line: {e}")
    return mappings


def advise_available(proc_maps: Path = Path("/proc/self/maps")) -> bool:
    """True when advise_paths() is usable."""
    if sys.platform in {"darwin", "win32"}:
        return False
    if not proc_maps.exists():
        return False
    try:
        libc = ctypes.CDLL(None)
        return hasattr(libc, "madvise")
    except (AttributeError, OSError, TypeError):
        return False


def advise_paths(
    paths: Iterable[Path], advice: int, proc_maps: Path = Path("/proc/self/maps")
) -> AdviseResults:
    """Apply madvise(advice) to this process's existing mappings of the paths.

    Advice ints are defined in mmap, e.g. mmap.MADV_RANDOM:
    https://docs.python.org/3/library/mmap.html#madv-constants

    Returns every path's result in input order. Paths to the same file share
    one result.

    Raises OSError if proc_maps can't be read.

    https://man7.org/linux/man-pages/man2/madvise.2.html
    """
    results: dict[Path, MappingResult] = {}

    # The maps file identifies files by (device, inode), not path.
    target_file_ids: dict[FileId, MappingResult] = {}
    for p in paths:
        try:
            st = p.stat()
        except OSError as e:
            results[p] = MappingResult(errno=e.errno)
            continue
        file_id = FileId(st.st_dev, st.st_ino)
        results[p] = target_file_ids.setdefault(file_id, MappingResult(stat=st))

    # Python's mmap.madvise() can't be used here. It only advises mappings the
    # mmap module created, not ones made in C/C++ like Arrow's memory_map in
    # HF Datasets.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    for mapping in read_memory_mappings(proc_maps):
        if mapping.id not in target_file_ids:
            continue

        result = target_file_ids[mapping.id]
        if libc.madvise(mapping.start, mapping.size, advice) == 0:
            result.num_bytes += mapping.size
        elif result.errno is None:  # don't overwrite the first error
            result.errno = ctypes.get_errno()
    return AdviseResults(
        paths=results,
        num_bytes=sum(result.num_bytes for result in target_file_ids.values()),
        num_files=len(target_file_ids),
    )
