import logging
import sys
from typing import Any, cast

from accelerate import Accelerator

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners import cross_validation
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Silence HTTP request logs from underlying libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> None:
    """Entry point for the evaluation pipeline.

    Usage::

        uv run run_evaluate.py <config.yaml> [<override.yaml> ...] [key=value ...]

    One or more YAML files may be supplied; later files override earlier ones
    for shared keys.  Any ``key=value`` arguments (containing ``=``) are
    treated as OmegaConf dotlist overrides applied after YAML merging.

    Examples::

        uv run modules/evaluate/src/run_evaluate.py modules/evaluate/configs/config.yaml
        uv run modules/evaluate/src/run_evaluate.py modules/evaluate/configs/config.yaml configs/experiment.yaml
        uv run modules/evaluate/src/run_evaluate.py modules/evaluate/configs/config.yaml dataset.base_path=/data
    """
    args = sys.argv[1:]
    if not args:
        print(
            "Usage: run_evaluate <config.yaml> [<override.yaml> ...] [key=value ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Separate yaml paths (no ``=``) from dotlist overrides (contain ``=``).
    config_paths = [a for a in args if "=" not in a]
    overrides = [a for a in args if "=" in a]

    if not config_paths:
        print("Error: at least one YAML config file must be provided.", file=sys.stderr)
        sys.exit(1)

    logger.info("=== Starting Evaluation Pipeline ===")

    config = load_and_parse(
        path=config_paths, model_cls=EvaluationConfig, overrides=overrides or None
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.steps.tune.accumulation_steps
    )

    logger.info(
        "Loaded config | workspace.model_name=%s | workspace.dataset_name=%s | workspace.task_type=%s | steps.mode=%s | steps.tune.hyperparameter_search=%s | steps.score.bootstrap_enabled=%s",
        config.workspace.model_name,
        config.workspace.dataset_name,
        config.workspace.task_type,
        config.steps.mode,
        config.steps.tune.hyperparameter_search.enabled,
        config.steps.score.bootstrap_enabled,
    )

    # Optionally start a Weights & Biases run so the steps' log_wandb() calls
    # (training loss, validation and test metrics) are recorded. Import lazily so
    # wandb is only required when actually enabled. Only the main process
    # initializes a run: under multi-process accelerate launches, every
    # rank would otherwise create its own duplicate wandb run.
    wandb_run = None
    if config.wandb.enabled and accelerator.is_main_process:
        import wandb

        wandb_api = cast(Any, wandb)
        wandb_run = wandb_api.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name
            or (
                f"{config.workspace.model_name}-{config.workspace.dataset_name}"
                f"-{config.variant}"
            ),
            tags=config.wandb.tags,
            config=config.model_dump(mode="json"),
        )

    # Seed everything for reproducibility
    logger.info("Seeding everything for reproducibility...")
    seed_everything(config.seed)

    if config.cross_validation.enabled:
        manifest_path = cross_validation.run_cross_validation(config, accelerator)
        logger.info("Cross-validation manifest written to %s", manifest_path)
    else:
        # Nest outputs under this run's variant (head type + mode + non-default
        # split) so sweeping probes for the same model+dataset doesn't
        # overwrite a previous variant's checkpoint/predictions/scores.
        workspace = get_evaluation_workspace(config.workspace, variant=config.variant)
        logger.info(
            "Initialized workspace: model %s dataset %s variant %s",
            workspace.model_name,
            workspace.dataset_name,
            config.variant,
        )
        run_pipeline(config, workspace, accelerator)

    logger.info("=== Evaluation Pipeline Completed Successfully ===")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
