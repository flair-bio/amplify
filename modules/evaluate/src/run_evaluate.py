import logging
import sys
from typing import Any, cast

from accelerate import Accelerator

from modules.evaluate.src.config import EvaluationConfig
from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.steps.predict import PredictStep
from modules.evaluate.src.steps.score import ScoreStep
from modules.evaluate.src.steps.tune import TuneStep
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
            name=config.wandb.run_name,
            tags=config.wandb.tags,
            config=config.model_dump(mode="json"),
        )

    # Seed everything for reproducibility
    logger.info("Seeding everything for reproducibility...")
    seed_everything(config.seed)

    # set up workspace, nested under this run's variant (head type + mode +
    # non-default split) so sweeping probes for the same model+dataset doesn't
    # overwrite a previous variant's checkpoint/predictions/scores.
    workspace = get_evaluation_workspace(config.workspace, variant=config.variant)

    logger.info(
        "Initialized workspace: model %s dataset %s variant %s (Class: %s)",
        workspace.model_name,
        workspace.dataset_name,
        config.variant,
        workspace.__class__.__name__,
    )

    # Execute steps in order

    logger.info("=== Starting Prepare Step ===")
    prepare_step = PrepareStep(config.steps.prepare)
    prepared_artifacts = prepare_step.run(
        workspace, seed=config.seed, accelerator=accelerator
    )
    logger.info("=== Completed Prepare Step ===")

    # Train in tune_probe or finetune mode. evaluate_as_is evaluates the task
    # head loaded by PrepareStep without entering TuneStep.
    mode = config.steps.mode
    if mode != "evaluate_as_is":
        logger.info("=== Fine-tuning model ===")
        tune_step = TuneStep(config.steps.tune, mode=mode)
        prepared_artifacts = tune_step.run(
            workspace, prepared_artifacts, seed=config.seed, accelerator=accelerator
        )
        logger.info("=== Completed Fine-tune Step ===")
    else:
        # evaluate_as_is returns a raw model from PrepareStep, so it still
        # needs accelerator.prepare() before Predict/Score can use it.
        prepared_artifacts.model = accelerator.prepare(prepared_artifacts.model)

    logger.info("=== Starting Predict Step ===")
    predict_step = PredictStep(config.steps.predict)
    tuned = mode != "evaluate_as_is"
    predict_result = predict_step.run(
        workspace,
        prepared_artifacts,
        seed=config.seed,
        accelerator=accelerator,
        # head_type doesn't apply to the "pretrained" (steps.tune never ran)
        # condition -- there's no trained head to name.
        head_type=config.steps.tune.head.head_type if tuned else None,
        variant=config.variant,
    )
    logger.info("=== Completed Predict Step ===")

    logger.info("=== Starting Score Step ===")
    score_step = ScoreStep(config.steps.score)
    # ScoreStep is a plain, single-process step, so only rank 0 needs to run
    # it. Passing PredictOutput avoids a disk round trip when the steps run in
    # the same process; ScoreStep still supports manifest-based reloading for
    # standalone or restarted scoring.
    if accelerator.is_main_process:
        result = score_step.run(
            workspace, seed=config.seed, predict_result=predict_result
        )
        logger.info("=== Completed Score Step ===")

        if result.summary_path is not None:
            logger.info(
                "Bootstrap confidence intervals written to %s",
                result.summary_path,
            )
    accelerator.wait_for_everyone()

    logger.info("=== Evaluation Pipeline Completed Successfully ===")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
