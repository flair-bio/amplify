"""Shared helpers for downloading source files to the dataset cache."""

from __future__ import annotations

import csv
import gzip
import logging
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl

from modules.evaluate.src.dataset_sourcing.config import (
    UniProtDatasetSourcingConfig,
    UniProtReviewStatus,
)

logger = logging.getLogger(__name__)
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
USER_AGENT = "flair-plm-dataset-sourcing"


def build_uniprot_query(
    organism_id: int | None,
    review_status: UniProtReviewStatus,
) -> str:
    """Build independent UniProt organism and Swiss-Prot/TrEMBL filters.

    Returns ``*`` (everything) when neither filter applies.
    """
    filters = []
    if organism_id is not None:
        filters.append(f"(organism_id:{organism_id})")
    if review_status == "reviewed":
        filters.append("(reviewed:true)")
    elif review_status == "unreviewed":
        filters.append("(reviewed:false)")
    return " AND ".join(filters) or "*"


def build_uniprot_tsv_url(
    fields: Iterable[str],
    organism_id: int | None,
    review_status: UniProtReviewStatus,
) -> str:
    """Build a UniProtKB stream URL; the full URL doubles as the cache identity."""
    params = {
        "compressed": "true",
        "fields": ",".join(fields),
        "format": "tsv",
        "query": build_uniprot_query(organism_id, review_status),
    }
    return f"{UNIPROT_STREAM_URL}?{urllib.parse.urlencode(params)}"


def download_uniprot_tsv(
    config: UniProtDatasetSourcingConfig,
    downloads_dir: Path,
    filename: str,
    fields: Iterable[str],
) -> Path:
    """Download a compressed UniProt TSV using the configured organism/review filters."""
    # Always request the filter columns so validate_uniprot_tsv can audit every row.
    requested_fields = list(dict.fromkeys([*fields, "organism_id", "reviewed"]))
    url = build_uniprot_tsv_url(
        requested_fields, config.organism_id, config.review_status
    )
    return download_url(
        url,
        downloads_dir / filename,
        force=config.force_download,
        cache_identity=url,
        validator=lambda path: validate_uniprot_tsv(path, config),
    )


def read_uniprot_tsv(
    path: Path, max_sequence_length: int | None = None
) -> pl.DataFrame:
    """Read a UniProt TSV, dropping sequences longer than *max_sequence_length*."""
    raw = pl.read_csv(path, separator="\t")
    if max_sequence_length is None:
        return raw
    sequence_column = next(col for col in raw.columns if col.lower() == "sequence")
    return raw.filter(
        pl.col(sequence_column).str.strip_chars().str.len_chars() <= max_sequence_length
    )


def validate_uniprot_tsv(path: Path, config: UniProtDatasetSourcingConfig) -> bool:
    """Check the actual source rows against the requested taxonomy/review filters."""
    # Sniff the gzip magic bytes so both gzipped and plain TSV files validate.
    with path.open("rb") as source:
        compressed = source.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else Path.open
    with opener(path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        required = {"Organism (ID)", "Reviewed"}
        if not required.issubset(reader.fieldnames or []):
            logger.error("UniProt TSV %s is missing filter-validation columns", path)
            return False
        for row_number, row in enumerate(reader, start=2):
            if (
                config.organism_id is not None
                and row["Organism (ID)"] != str(config.organism_id)
            ) or (
                config.review_status != "all"
                and row["Reviewed"] != config.review_status
            ):
                logger.error(
                    "UniProt TSV %s row %s violates query %s: organism=%s reviewed=%s",
                    path,
                    row_number,
                    build_uniprot_query(config.organism_id, config.review_status),
                    row["Organism (ID)"],
                    row["Reviewed"],
                )
                return False
    return True


def download_url(
    url: str,
    destination: Path,
    force: bool = False,
    validator: Callable[[Path], bool] | None = None,
    cache_identity: str | None = None,
) -> Path:
    """Download a URL atomically, optionally requiring cache provenance to match.

    *cache_identity* is stored in a ``<file>.url`` sidecar; a cached file is reused
    only if the sidecar matches and *validator* (if any) passes. Otherwise it is
    re-downloaded, and a file that fails validation is never moved into place.
    """
    identity_path = destination.with_suffix(destination.suffix + ".url")
    identity_matches = cache_identity is None or (
        identity_path.exists()
        and identity_path.read_text(encoding="utf-8") == cache_identity
    )
    if (
        destination.exists()
        and not force
        and identity_matches
        and (validator is None or validator(destination))
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    if cache_identity is not None:
        identity_path.unlink(missing_ok=True)
    logger.info("Downloading %s to %s", url, destination)
    # Some hosts (e.g. purl.obolibrary.org) return 403 for urllib's default User-Agent.
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", USER_AGENT)]
    urllib.request.install_opener(opener)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        if validator is not None and not validator(temporary_path):
            raise ValueError(f"Downloaded file failed validation: {temporary_path}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    if cache_identity is not None:
        identity_path.write_text(cache_identity, encoding="utf-8")
    return destination
