import ctypes
import errno
import logging
import mmap
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.core.utils.madvise import (
    MADV_RANDOM,
    FileId,
    MappingResult,
    MemoryMapping,
    advise_paths,
)

TEST_MAP_START = 0x7F0A1C000000
TEST_MAP_END = TEST_MAP_START + mmap.PAGESIZE


def _file(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.touch()
    return path


def _maps_line(path: Path, start: int, pages: int = 1, shown: str | None = None) -> str:
    """A maps line for ``path``'s device/inode, printing ``shown`` as its path."""
    st = path.stat()
    dev = f"{os.major(st.st_dev):x}:{os.minor(st.st_dev):x}"
    end = start + pages * mmap.PAGESIZE
    return f"{start:x}-{end:x} r--s 00000000 {dev} {st.st_ino} {shown or path}\n"


def _proc_maps(tmp_path: Path, *lines: str) -> Path:
    proc_maps = tmp_path / "maps"
    proc_maps.write_text("".join(lines))
    return proc_maps


@pytest.fixture
def madvise(monkeypatch):
    """Replace libc ``madvise``: records calls, fails where ``errnos[start]`` is set."""
    fake = SimpleNamespace(calls=[], errnos={}, last=0)

    def _madvise(start, length, advice):
        fake.calls.append((start, length, advice))
        fake.last = fake.errnos.get(start, 0)
        return -1 if fake.last else 0

    monkeypatch.setattr(
        ctypes, "CDLL", lambda *a, **k: SimpleNamespace(madvise=_madvise)
    )
    monkeypatch.setattr(ctypes, "get_errno", lambda: fake.last)
    return fake


class TestAdvisePaths:
    def test_reports_every_path_in_order(self, tmp_path, madvise):
        target = _file(tmp_path, "target.bin")
        other = _file(tmp_path, "other.bin")
        unmapped = _file(tmp_path, "unmapped.bin")
        missing = tmp_path / "missing.bin"
        alias = tmp_path / "alias.bin"
        alias.symlink_to(target)
        proc_maps = _proc_maps(
            tmp_path,
            _maps_line(target, 1 << 20, pages=2),
            _maps_line(other, 2 << 20),
            "300000-301000 rw-p 00000000 00:00 0 [heap]\n",
            # Matched by inode, not the printed path: e.g. opened via a symlink...
            _maps_line(target, 5 << 20, shown="/elsewhere/renamed.bin"),
            # ...and a different file printed under the target's path is not.
            _maps_line(other, 6 << 20, shown=str(target)),
        )

        results = advise_paths(
            [missing, target, alias, unmapped], mmap.MADV_SEQUENTIAL, proc_maps
        )

        assert list(results.paths) == [missing, target, alias, unmapped]
        advised = MappingResult(target.stat(), 3 * mmap.PAGESIZE)
        assert results.paths == {
            missing: MappingResult(errno=errno.ENOENT),
            target: advised,
            alias: advised,
            unmapped: MappingResult(unmapped.stat()),
        }
        # The alias shares the target's result, so its bytes count once.
        assert (results.num_bytes, results.num_files) == (3 * mmap.PAGESIZE, 2)
        assert madvise.calls == [
            (1 << 20, 2 * mmap.PAGESIZE, mmap.MADV_SEQUENTIAL),
            (5 << 20, mmap.PAGESIZE, mmap.MADV_SEQUENTIAL),
        ]

    def test_skips_unparseable_lines(self, tmp_path, madvise, caplog):
        target = _file(tmp_path, "target.bin")
        proc_maps = _proc_maps(tmp_path, "garbage\n", _maps_line(target, 1 << 20))
        with caplog.at_level(logging.DEBUG):
            results = advise_paths([target], MADV_RANDOM, proc_maps)
        assert results.paths == {target: MappingResult(target.stat(), mmap.PAGESIZE)}
        assert [r.levelno for r in caplog.records] == [logging.DEBUG]

    def test_madvise_errors_are_returned(self, tmp_path, madvise):
        target = _file(tmp_path, "target.bin")
        proc_maps = _proc_maps(
            tmp_path,
            _maps_line(target, 1 << 20),
            _maps_line(target, 2 << 20),
            _maps_line(target, 3 << 20),
        )
        madvise.errnos = {1 << 20: errno.EPERM, 2 << 20: errno.ENOMEM}

        results = advise_paths([target], MADV_RANDOM, proc_maps)

        # The first error is kept, and later mappings are still advised.
        assert results.paths == {
            target: MappingResult(target.stat(), mmap.PAGESIZE, errno.EPERM)
        }
        assert len(madvise.calls) == 3


class TestMemoryMapping:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            (
                (
                    "7f0a1c000000-7f0a1c004000 r--s 00000000 fd:1a 16840349346116965730"
                    "   /data/a b.arrow\n"
                ),
                MemoryMapping(
                    start=0x7F0A1C000000,
                    end=0x7F0A1C004000,
                    device=os.makedev(0xFD, 0x1A),
                    inode=16840349346116965730,
                    path=Path("/data/a b.arrow"),
                ),
            ),
            (
                f"{TEST_MAP_START:x}-{TEST_MAP_END:x} rw-p 00000000 00:00 0\n",
                MemoryMapping(
                    start=TEST_MAP_START,
                    end=TEST_MAP_END,
                    device=0,
                    inode=0,
                    path=None,
                ),
            ),
        ],
        ids=["file-backed", "anonymous"],
    )
    def test_parses_a_line(self, line, expected):
        mapping = MemoryMapping.from_line(line)
        assert mapping == expected
        assert mapping.id == FileId(expected.device, expected.inode)

    @pytest.mark.parametrize(
        "line",
        [
            "not a maps line",
            "7f0a1c000000-7f0a1c001000 r--p 00000000 0033 0",
            "7f0a1c001000-7f0a1c000000 r--p 00000000 00:00 0",
            "7f0a1c000001-7f0a1c001000 r--p 00000000 00:00 0",
        ],
    )
    def test_rejects_invalid_lines(self, line):
        with pytest.raises(ValueError):
            MemoryMapping.from_line(line)
