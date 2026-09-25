from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict


class DatasetConfig(BaseModel):
    """Shared config for every dataset type."""

    model_config = ConfigDict(extra="forbid")

    # dataset name
    name: str
    # base path where the dataset is stored
    base_path: str
    # whether to use the test mode
    test_mode: bool = True
    # path for downloaded files
    download_path: str | None = None
    # path for temporary files
    tmp_path: str | None = None
    # list of URLs to download the dataset from
    source_urls: list[str] | None = None


class Dataset:
    """Generic dataset class."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.name = config.name

        base = Path(config.base_path)
        self.dataset_root = base / self.name

        self.train_path = self.dataset_root / "train"
        self.tmp_path = (
            Path(config.tmp_path) if config.tmp_path else self.dataset_root / "tmp"
        )
        self.download_path = (
            Path(config.download_path)
            if config.download_path
            else self.dataset_root / "download"
        )
        self.stats_path = self.dataset_root / "stats"

        self.source_urls: list[str] = config.source_urls or []

        self._all_paths = [
            self.train_path,
            self.tmp_path,
            self.download_path,
            self.stats_path,
        ]

    def setup_directories(self) -> None:
        """Create all required directories on disk."""
        for path in self._all_paths:
            path.mkdir(parents=True, exist_ok=True)
