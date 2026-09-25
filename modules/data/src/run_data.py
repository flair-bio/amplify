import logging
import os
import sys

from modules.data.src.config import DataPipelineConfig
from modules.core.utils.config_loader import load_and_parse
from modules.data.src.dataset.dataset import Dataset
from modules.data.src.steps.download import DownloadStep
from modules.data.src.steps.preprocess import PreprocessStep
from modules.data.src.steps.cluster import ClusterStep
from modules.data.src.steps.score import ScoreStep
from modules.data.src.steps.assemble import AssembleStep
from modules.data.src.steps.upload import UploadStep
from modules.data.src.steps.stats import StatsStep


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# In distributed launches, keep detailed lifecycle logs on rank 0 only.
if str(os.environ.get("LOCAL_RANK", "0")) != "0":
    logging.getLogger().setLevel(logging.WARNING)


def main() -> None:
    """Entry point for the data pipeline.

    Usage::

        uv run run_data.py <config.yaml> [<override.yaml> ...] [key=value ...]

    One or more YAML files may be supplied; later files override earlier ones
    for shared keys.  Any ``key=value`` arguments (containing ``=``) are
    treated as OmegaConf dotlist overrides applied after YAML merging.

    Examples::

        uv run modules/data/src/run_data.py modules/data/configs/config.yaml
        uv run modules/data/src/run_data.py modules/data/configs/config.yaml configs/experiment.yaml
        uv run modules/data/src/run_data.py modules/data/configs/config.yaml dataset.base_path=/data
    """
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: run_data <config.yaml> [<override.yaml> ...] [key=value ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Separate yaml paths (no ``=``) from dotlist overrides (contain ``=``).
    config_paths = [a for a in args if "=" not in a]
    overrides = [a for a in args if "=" in a]

    if not config_paths:
        print("Error: at least one YAML config file must be provided.", file=sys.stderr)
        sys.exit(1)

    logger.info("=== Starting Data Pipeline Runner ===")

    config = load_and_parse(
        path=config_paths, model_cls=DataPipelineConfig, overrides=overrides or None
    )

    logger.info(
        "Loaded config | dataset.name=%s | steps.download=%s | steps.preprocess=%s | steps.cluster=%s | steps.score=%s | steps.assemble=%s | steps.upload=%s | steps.stats=%s",
        config.dataset.name,
        config.steps.download.enabled,
        config.steps.preprocess.enabled,
        config.steps.cluster.enabled,
        config.steps.score.enabled,
        config.steps.assemble.enabled,
        config.steps.upload.enabled,
        config.steps.stats.enabled,
    )

    # build dataset
    dataset = Dataset(config.dataset)
    logger.info(
        "Initialized dataset: %s (Class: %s)", dataset.name, dataset.__class__.__name__
    )

    # Execute steps in order
    if config.steps.download.enabled:
        logger.info("=== Starting Download Step ===")
        download_step = DownloadStep(config.steps.download)
        download_step.run(dataset)
        logger.info("=== Completed Download Step ===")
    if config.steps.preprocess.enabled:
        logger.info("=== Starting Preprocess Step ===")
        preprocess_step = PreprocessStep(config.steps.preprocess)
        preprocess_step.run(dataset)
        logger.info("=== Completed Preprocess Step ===")
    if config.steps.cluster.enabled:
        logger.info("=== Starting Cluster Step ===")
        cluster_step = ClusterStep(config.steps.cluster)
        cluster_step.run(dataset)
        logger.info("=== Completed Cluster Step ===")
    if config.steps.score.enabled:
        logger.info("=== Starting Score Step ===")
        score_step = ScoreStep(config.steps.score)
        score_step.run(dataset)
        logger.info("=== Completed Score Step ===")
    if config.steps.assemble.enabled:
        logger.info("=== Starting Assemble Step ===")
        assemble_step = AssembleStep(config.steps.assemble)
        assemble_step.run(dataset)
        logger.info("=== Completed Assemble Step ===")
    if config.steps.upload.enabled:
        logger.info("=== Starting Upload Step ===")
        upload_step = UploadStep(config.steps.upload)
        upload_step.run(dataset)
        logger.info("=== Completed Upload Step ===")
    if config.steps.stats.enabled:
        logger.info("=== Starting Stats Step ===")
        stats_step = StatsStep(config.steps.stats)
        stats_step.run(dataset)
        logger.info("=== Completed Stats Step ===")
    logger.info("=== Data Pipeline Completed ===")


if __name__ == "__main__":
    main()
