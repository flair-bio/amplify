"""Top-level data pipeline config.

Composes per-component config classes into a single validated schema.
Use :func:`modules.core.utils.config_loader.load_and_parse` to build an
instance from YAML files and optional CLI dotlist overrides.
"""

from pydantic import BaseModel, Field, ConfigDict

from modules.data.src.dataset.dataset import DatasetConfig

from modules.data.src.steps.download import DownloadConfig
from modules.data.src.steps.preprocess import PreprocessConfig
from modules.data.src.steps.cluster import ClusterConfig
from modules.data.src.steps.score import ScoreConfig
from modules.data.src.steps.assemble import AssembleConfig
from modules.data.src.steps.upload import UploadConfig
from modules.data.src.steps.stats import StatsConfig


class StepsConfig(BaseModel):
    """Composition of all pipeline step configs."""

    model_config = ConfigDict(extra="forbid")

    download: DownloadConfig = Field(default_factory=DownloadConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    score: ScoreConfig = Field(default_factory=ScoreConfig)
    assemble: AssembleConfig = Field(default_factory=AssembleConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)


class DataPipelineConfig(BaseModel):
    """Overall config. for the data pipeline."""

    model_config = ConfigDict(extra="forbid")

    # dataset config.
    dataset: DatasetConfig

    # steps config.
    steps: StepsConfig = Field(default_factory=StepsConfig)
