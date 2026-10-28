import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import Accelerator

from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.suites import (
    EvaluationJob,
    SuiteError,
    expand_suite,
    model_label,
    resolve_model_collection,
    resolve_model_paths,
    select_jobs,
)
from modules.evaluate.src.utils.config_loader import load_evaluation_config
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


LEGACY_HINT = (
    "Positional YAML configs are no longer accepted. Select a suite and model, e.g.\n"
    "  run_evaluate --suite calibration --model amplify_120m [--config FILE] [key=value ...]\n"
    "Use --config FILE for extra YAML overlays."
)


@dataclass(frozen=True)
class PlannedRun:
    """One concrete run: a suite job bound to a model (or to baselines only)."""

    job: EvaluationJob
    model_label: str | None
    config_paths: tuple[Path, ...]
    overrides: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_evaluate",
        description="Run evaluation suites for one model, a model collection, or baselines.",
    )
    parser.add_argument("--suite", required=True, help="Suite name or YAML path.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", help="Model config name or YAML path.")
    mode.add_argument("--model-collection", help="Model collection name or YAML path.")
    mode.add_argument(
        "--baselines", action="store_true", help="Run shared baselines (no model)."
    )
    parser.add_argument("--model-name", help="Label override for a single model.")
    parser.add_argument("--predictor", help="Run only jobs with this predictor.")
    parser.add_argument("--data-root", help="Not implemented yet; rejected.")
    parser.add_argument("--output-root", help="Output directory (workspace.output_dir).")
    parser.add_argument("--seed", type=int, help="Evaluation seed.")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        dest="config_files",
        metavar="OVERRIDE_YAML",
        help="Extra YAML overlay; repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compose and validate jobs without loading models or data.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace compatible model outputs."
    )
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if any("=" not in token for token in args.overrides):
        raise SystemExit(LEGACY_HINT)
    if args.data_root is not None:
        raise SystemExit(
            "--data-root is not implemented yet; use dataset configs or --config."
        )
    if args.baselines and args.overwrite:
        raise SystemExit(
            "--overwrite applies to model runs only; baseline artifacts are validated "
            "and reused by --baselines."
        )
    if args.model_name is not None and args.model is None:
        raise SystemExit("--model-name is only valid with --model (single model).")


def plan_runs(args: argparse.Namespace) -> list[PlannedRun]:
    """Expand suite, selectors and model inputs into concrete runs (offline)."""
    jobs = expand_suite(args.suite)
    if args.predictor is not None:
        jobs = select_jobs(jobs, predictor=args.predictor)
    extra_files = [Path(path) for path in args.config_files]
    for path in extra_files:
        if not path.is_file():
            raise SystemExit(f"--config file not found: {path}")
    shared: list[str] = []
    if args.seed is not None:
        shared.append(f"seed={args.seed}")
    if args.output_root is not None:
        shared.append(f"workspace.output_dir={args.output_root}")
    if args.model_name is not None:
        shared.append(f"workspace.model_name={args.model_name}")

    models: list[Path | None]
    if args.baselines:
        models = [None]
        # Baselines are shared per task/dataset; predictors do not apply.
        seen: set[tuple[str, str]] = set()
        unique: list[EvaluationJob] = []
        for job in jobs:
            key = (job.task, job.dataset)
            if key not in seen:
                seen.add(key)
                unique.append(replace(job, predictor=None))
        jobs = unique
    elif args.model_collection is not None:
        models = list(resolve_model_collection(args.model_collection))
    else:
        models = list(resolve_model_paths(args.model))

    runs: list[PlannedRun] = []
    for model in models:
        for job in jobs:
            runs.append(
                PlannedRun(
                    job=job,
                    model_label=model_label(model) if model is not None else None,
                    config_paths=tuple(job.config_paths(model)) + tuple(extra_files),
                    overrides=tuple(job.overrides)
                    + tuple(shared)
                    + tuple(args.overrides),
                )
            )
    return runs


def _compose_for_dry_run(run: PlannedRun, baselines: bool) -> str:
    paths = [str(path) for path in run.config_paths]
    overrides = list(run.overrides)
    if baselines:
        from modules.evaluate.src.runners.baselines import load_baseline_config

        config = load_baseline_config(paths, overrides or None)
        workspace = config.workspace
        return (
            f"task={run.job.task} dataset={workspace.dataset_name} "
            f"predictor={run.job.predictor or '-'} model=- seed={config.seed}"
        )
    composed = load_evaluation_config(path=paths, overrides=overrides or None)
    return (
        f"task={composed.workspace.task_type} dataset={composed.workspace.dataset_name} "
        f"predictor={run.job.predictor or '-'} model={run.model_label or '-'} "
        f"seed={composed.seed} output_dir={composed.workspace.output_dir}"
    )


def _run_baselines(run: PlannedRun) -> None:
    from modules.evaluate.src.runners.baselines import (
        load_baseline_config,
        run_baselines,
    )

    baseline_config = load_baseline_config(
        [str(path) for path in run.config_paths], list(run.overrides) or None
    )
    for test, artifacts in run_baselines(baseline_config).items():
        for name, path in artifacts.items():
            logger.info("Baseline %s / %s: %s", test, name, path)


def _run_model_evaluation(
    config_paths: list[str], overrides: list[str], overwrite: bool, task: str
) -> None:
    """Run one model/dataset/predictor job end to end (legacy single-run body)."""
    logger.info("=== Starting Evaluation Pipeline ===")

    config = load_evaluation_config(path=config_paths, overrides=overrides or None)
    torch.set_float32_matmul_precision("high" if config.tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = config.tf32
    torch.backends.cudnn.allow_tf32 = config.tf32
    logger.info("CUDA TF32 allowed: %s", config.tf32)
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

    # Nest outputs under this run's variant (head type + mode + non-default
    # split) so sweeping probes for the same model+dataset doesn't
    # overwrite a previous variant's checkpoint/predictions/scores.
    workspace = get_evaluation_workspace(
        config.workspace, variant=config.variant, seed=config.seed
    )
    logger.info(
        "Initialized workspace: model %s dataset %s variant %s",
        workspace.model_name,
        workspace.dataset_name,
        config.variant,
    )
    run_pipeline(config, workspace, accelerator, overwrite=overwrite)

    logger.info("=== Evaluation Pipeline Completed Successfully ===")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
