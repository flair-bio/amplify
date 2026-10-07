from pathlib import Path
from unittest.mock import patch

import pytest

from modules.evaluate.src.dataset_sourcing import downloads
from modules.evaluate.src.dataset_sourcing.downloads import (
    build_uniprot_query,
    download_url,
)


@pytest.mark.parametrize(
    ("organism_id", "review_status", "expected"),
    [
        (None, "all", "*"),
        (9606, "all", "(organism_id:9606)"),
        (None, "reviewed", "(reviewed:true)"),
        (559292, "unreviewed", "(organism_id:559292) AND (reviewed:false)"),
    ],
)
def test_build_uniprot_query(organism_id, review_status, expected):
    assert build_uniprot_query(organism_id, review_status) == expected


def test_download_url_reuses_cached_file_without_validation(tmp_path):
    destination = tmp_path / "cached.tsv"
    destination.write_text("cached", encoding="utf-8")

    with patch.object(downloads.urllib.request, "urlretrieve") as urlretrieve:
        result = download_url("https://example.test/file", destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "cached"
    urlretrieve.assert_not_called()


def test_download_url_redownloads_invalid_cached_file(tmp_path):
    destination = tmp_path / "archive.tar.gz"
    destination.write_text("invalid", encoding="utf-8")

    def retrieve(_url: str, target: str | Path) -> None:
        Path(target).write_text("valid", encoding="utf-8")

    with patch.object(downloads.urllib.request, "urlretrieve", side_effect=retrieve):
        result = download_url(
            "https://example.test/archive",
            destination,
            validator=lambda path: path.read_text(encoding="utf-8") == "valid",
        )

    assert result == destination
    assert destination.read_text(encoding="utf-8") == "valid"


def test_download_url_validates_temporary_file_before_replacing_destination(tmp_path):
    destination = tmp_path / "archive.tar.gz"
    destination.write_text("existing", encoding="utf-8")

    def retrieve(_url: str, target: str | Path) -> None:
        Path(target).write_text("invalid", encoding="utf-8")

    with (
        patch.object(downloads.urllib.request, "urlretrieve", side_effect=retrieve),
        pytest.raises(ValueError, match="failed validation"),
    ):
        download_url(
            "https://example.test/archive",
            destination,
            force=True,
            validator=lambda path: path.read_text(encoding="utf-8") == "valid",
        )

    assert destination.read_text(encoding="utf-8") == "existing"
    assert not destination.with_suffix(destination.suffix + ".tmp").exists()
