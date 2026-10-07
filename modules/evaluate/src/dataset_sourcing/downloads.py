"""Shared helpers for downloading source files to the dataset cache."""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from modules.evaluate.src.dataset_sourcing.config import (
    UniProtDatasetSourcingConfig,
    UniProtReviewStatus,
)

logger = logging.getLogger(__name__)
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"


def build_uniprot_query(
    organism_id: int | None,
    review_status: UniProtReviewStatus,
) -> str:
    """Build independent UniProt organism and Swiss-Prot/TrEMBL filters."""
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
    return download_url(
        build_uniprot_tsv_url(fields, config.organism_id, config.review_status),
        downloads_dir / filename,
        force=config.force_download,
    )


def download_url(
    url: str,
    destination: Path,
    force: bool = False,
    validator: Callable[[Path], bool] | None = None,
) -> Path:
    """Download a URL atomically, reusing a valid cached file when possible."""
    if (
        destination.exists()
        and not force
        and (validator is None or validator(destination))
    ):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    logger.info("Downloading %s to %s", url, destination)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        if validator is not None and not validator(temporary_path):
            raise ValueError(f"Downloaded file failed validation: {temporary_path}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination
