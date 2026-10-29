"""Atomic publication and bounded, cross-process cache coordination."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, sleep


@contextmanager
def atomic_output(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def cache_lock(
    path: Path, *, timeout: float = 300, poll_interval: float = 0.1
) -> Iterator[None]:
    """Use atomic mkdir, not advisory locks; never reclaim another job's lease."""
    directory = path.with_name(f"{path.name}.d")
    directory.parent.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + timeout
    while True:
        try:
            directory.mkdir()
            break
        except FileExistsError:
            if monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for {directory}. If its owning job has "
                    "terminated, remove the lock directory before retrying."
                ) from None
            sleep(poll_interval)
    owner = directory / "owner"
    try:
        owner.write_text(f"{socket.gethostname()}:{os.getpid()}\n")
        yield
    finally:
        owner.unlink(missing_ok=True)
        directory.rmdir()
