import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import download_with_aria2c

logger = logging.getLogger(__name__)


class DownloadConfig(BaseModel):
    """Config. for the download step."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overwrite: bool = False
    split: int = 16
    max_connections_per_server: int = 16
    min_split_size: str = "1M"
    max_concurrent_downloads: int = 1


class DownloadStep:
    def __init__(self, config: DownloadConfig) -> None:
        self.config = config

    def run(self, dataset: Dataset) -> None:
        """Download all files declared in ``dataset.source_urls``."""
        dataset.setup_directories()

        if not dataset.source_urls:
            logger.info(
                f"No source URLs provided for dataset {dataset.name}. Skipping download."
            )
            return

        urls_to_download = [
            url
            for url in dataset.source_urls
            if self.config.overwrite
            or not (dataset.download_path / Path(urlparse(url).path).name).exists()
        ]

        if not urls_to_download:
            logger.info(
                f"All files for {dataset.name} already exist and overwrite is disabled. Skipping download."
            )
            return

        is_multiple = len(urls_to_download) > 1
        download_with_aria2c(
            urls=urls_to_download,
            download_dir=str(dataset.download_path),
            is_multiple=is_multiple,
            split=self.config.split,
            max_connections=self.config.max_connections_per_server,
            min_split_size=self.config.min_split_size,
            max_concurrent=self.config.max_concurrent_downloads,
        )
