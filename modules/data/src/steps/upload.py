import logging
import os

from huggingface_hub import get_token
from pydantic import BaseModel, ConfigDict

from modules.core.utils.hf_upload import upload_to_hf
from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import resolve_path

logger = logging.getLogger(__name__)


class UploadConfig(BaseModel):
    """Config. for uploading processed parquet artifacts to Hugging Face Hub."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    repo_id: str | None = None
    private: bool = True
    split: str = "train"
    source_override: str | None = None
    revision: str | None = None
    commit_message: str = "Upload sharded parquet dataset"


class UploadStep:
    def __init__(self, config: UploadConfig) -> None:
        self.config = config

    def _resolve_token(self) -> str:
        # Prefer explicit env vars, then fall back to the token cached locally
        # by `hf auth login` (what `hf auth list` reads from).
        token = (
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or get_token()
        )
        if not token:
            raise ValueError(
                "Hugging Face token is required: set HF_TOKEN or "
                "HUGGINGFACE_HUB_TOKEN, or run `hf auth login`."
            )
        return token

    def run(self, dataset: Dataset) -> None:
        dataset.setup_directories()

        if not self.config.repo_id:
            raise ValueError("steps.upload.repo_id must be set when upload is enabled")

        # Upload only assembled artifacts produced by AssembleStep, unless
        # explicitly overridden. Single-file mode: {name}_assembled.parquet
        override = (
            os.path.expanduser(self.config.source_override)
            if self.config.source_override
            else None
        )
        default_file = dataset.train_path / f"{dataset.name}_assembled.parquet"
        source = resolve_path(
            override, default_file, "Upload source", required=bool(override)
        )

        if source is None:
            # Sharded mode: tolerate any {name}_assembled* directory of parquet
            # shards (e.g. {name}_assembled or {name}_assembled_shards).
            for candidate in sorted(
                dataset.train_path.glob(f"{dataset.name}_assembled*")
            ):
                if candidate.is_dir() and any(candidate.glob("*.parquet")):
                    source = candidate
                    break

        if source is None:
            raise FileNotFoundError(
                "Could not find assembled parquet output for upload. "
                "Run assemble step first or set steps.upload.source_override."
            )

        # Dataset-specific validation logic
        if source.is_dir() and not any(source.rglob("*.parquet")):
            raise ValueError(f"No parquet shards found in upload directory: {source}")
        if source.is_file() and source.suffix != ".parquet":
            raise ValueError(f"Upload source file must be parquet: {source}")

        # Delegate the actual heavy lifting to our core utility
        upload_to_hf(
            folder_path=source,
            repo_id=self.config.repo_id,
            repo_type="dataset",
            path_in_repo=self.config.split,
            token=self._resolve_token(),
            revision=self.config.revision,
            commit_message=self.config.commit_message,
            private=self.config.private,
            allow_patterns=["**/*.parquet", "*.parquet"],
        )

        logger.info(
            "Upload completed: %s -> https://huggingface.co/datasets/%s",
            source,
            self.config.repo_id,
        )
