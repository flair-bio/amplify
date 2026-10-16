import gzip
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pytest

from modules.evaluate.scripts.dataset_sourcing.source_interpro import (
    SourceInterProConfig,
    download_sources,
)
from modules.evaluate.src.dataset_sourcing import downloads
from modules.evaluate.src.dataset_sourcing.config import UniProtDatasetSourcingConfig
from modules.evaluate.src.dataset_sourcing.downloads import (
    build_uniprot_query,
    download_url,
    download_uniprot_tsv,
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


def test_download_uniprot_tsv_does_not_reuse_cache_for_a_different_query(tmp_path):
    all_organisms = UniProtDatasetSourcingConfig(
        datasets=["test"], organism_id=None, review_status="reviewed"
    )
    human = UniProtDatasetSourcingConfig(
        datasets=["test"], organism_id=9606, review_status="reviewed"
    )
    (tmp_path / "uniprot.tsv.gz").write_text("untracked cache", encoding="utf-8")
    calls = []

    def retrieve(url: str, target: str | Path) -> None:
        calls.append(url)
        query = parse_qs(urlparse(url).query)["query"][0]
        organism = 9606 if "organism_id:9606" in query else 10090
        with gzip.open(target, "wt", encoding="utf-8") as source:
            source.write(f"Entry\tOrganism (ID)\tReviewed\nP1\t{organism}\treviewed\n")

    with patch.object(downloads.urllib.request, "urlretrieve", side_effect=retrieve):
        first = download_uniprot_tsv(
            all_organisms, tmp_path, "uniprot.tsv.gz", ["accession"]
        )
        second = download_uniprot_tsv(
            human, tmp_path, "uniprot.tsv.gz", ["accession"]
        )
        third = download_uniprot_tsv(
            human, tmp_path, "uniprot.tsv.gz", ["accession"]
        )

    assert first == second == third
    assert len(calls) == 2
    assert calls[0] != calls[1]
    with gzip.open(third, "rt", encoding="utf-8") as source:
        assert "P1\t9606\treviewed" in source.read()


@pytest.mark.parametrize(
    "row",
    ["P1\t10090\treviewed", "P1\t9606\tunreviewed"],
)
def test_download_uniprot_tsv_rejects_rows_outside_requested_filters(tmp_path, row):
    config = UniProtDatasetSourcingConfig(
        datasets=["test"], organism_id=9606, review_status="reviewed"
    )

    def retrieve(_url: str, target: str | Path) -> None:
        with gzip.open(target, "wt", encoding="utf-8") as source:
            source.write(f"Entry\tOrganism (ID)\tReviewed\n{row}\n")

    with (
        patch.object(downloads.urllib.request, "urlretrieve", side_effect=retrieve),
        pytest.raises(ValueError, match="failed validation"),
    ):
        download_uniprot_tsv(config, tmp_path, "uniprot.tsv.gz", ["accession"])

    assert not (tmp_path / "uniprot.tsv.gz").exists()
    assert not (tmp_path / "uniprot.tsv.gz.url").exists()


def test_download_uniprot_tsv_checks_cached_contents_even_with_matching_url(tmp_path):
    config = UniProtDatasetSourcingConfig(
        datasets=["test"], organism_id=9606, review_status="reviewed"
    )
    destination = tmp_path / "uniprot.tsv.gz"
    destination.write_text(
        "Entry\tOrganism (ID)\tReviewed\nP1\t10090\treviewed\n", encoding="utf-8"
    )
    url = downloads.build_uniprot_tsv_url(
        ["accession", "organism_id", "reviewed"], 9606, "reviewed"
    )
    destination.with_suffix(".gz.url").write_text(url, encoding="utf-8")

    def retrieve(_url: str, target: str | Path) -> None:
        Path(target).write_text(
            "Entry\tOrganism (ID)\tReviewed\nP1\t9606\treviewed\n", encoding="utf-8"
        )

    with patch.object(
        downloads.urllib.request, "urlretrieve", side_effect=retrieve
    ) as urlretrieve:
        download_uniprot_tsv(config, tmp_path, "uniprot.tsv.gz", ["accession"])

    urlretrieve.assert_called_once()
    assert "P1\t9606\treviewed" in destination.read_text(encoding="utf-8")


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


def test_interpro_downloads_default_to_reviewed_human_and_validate_rows(tmp_path):
    config = SourceInterProConfig()
    assert config.organism_id == 9606
    assert config.review_status == "reviewed"
    calls = []

    def retrieve(url: str, target: str | Path) -> None:
        calls.append(url)
        if url.startswith(downloads.UNIPROT_STREAM_URL):
            with gzip.open(target, "wt", encoding="utf-8") as source:
                source.write(
                    "Entry\tSequence\tInterPro\tOrganism (ID)\tReviewed\n"
                    "P1\tAAAA\tIPR000001\t9606\treviewed\n"
                )
        else:
            Path(target).write_text(
                "ENTRY_AC\tENTRY_TYPE\tENTRY_NAME\n"
                "IPR000001\tBinding_site\tExample\n",
                encoding="utf-8",
            )

    with patch.object(downloads.urllib.request, "urlretrieve", side_effect=retrieve):
        uniprot_path, entry_list_path = download_sources(config, tmp_path)

    request = parse_qs(urlparse(calls[0]).query)
    assert request["query"] == ["(organism_id:9606) AND (reviewed:true)"]
    assert {"organism_id", "reviewed"}.issubset(request["fields"][0].split(","))
    assert uniprot_path.exists()
    assert entry_list_path.exists()


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
