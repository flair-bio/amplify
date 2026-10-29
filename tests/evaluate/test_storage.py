"""Failure-safe publication and independent-writer exclusion."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest

from modules.evaluate.src.utils.storage import atomic_output, cache_lock


def _increment(path: Path) -> None:
    with cache_lock(path.with_suffix(".lock"), timeout=10, poll_interval=0.005):
        value = int(path.read_text())
        with atomic_output(path) as temporary:
            temporary.write_text(str(value + 1))


def test_atomic_output_retains_previous_file_after_failure(tmp_path):
    path = tmp_path / "output"
    path.write_text("valid")
    with pytest.raises(RuntimeError), atomic_output(path) as temporary:
        temporary.write_text("partial")
        raise RuntimeError("extraction failed")
    assert path.read_text() == "valid"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_output_uses_independent_temporary_paths(tmp_path):
    path = tmp_path / "output"
    with atomic_output(path) as first, atomic_output(path) as second:
        assert first != second
        first.write_text("first")
        second.write_text("second")
    assert path.read_text() == "first"
    assert list(tmp_path.iterdir()) == [path]


def test_cache_lock_timeout_does_not_remove_existing_owner(tmp_path):
    path = tmp_path / ".lock"
    directory = tmp_path / ".lock.d"
    with cache_lock(path):
        owner = (directory / "owner").read_text()
        with (
            pytest.raises(TimeoutError, match="owning job"),
            cache_lock(path, timeout=0),
        ):
            pytest.fail("another writer acquired the lease")
        assert (directory / "owner").read_text() == owner
    assert not directory.exists()


def test_cache_lock_releases_after_failed_extraction(tmp_path):
    path = tmp_path / ".lock"
    with pytest.raises(RuntimeError), cache_lock(path):
        raise RuntimeError("extraction failed")
    with cache_lock(path, timeout=0):
        pass


def test_independent_process_writers_do_not_lose_updates(tmp_path):
    path = tmp_path / "counter"
    path.write_text("0")
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        list(pool.map(_increment, [path] * 8))
    assert path.read_text() == "8"
    assert list(tmp_path.iterdir()) == [path]
