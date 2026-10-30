import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import Accelerator

from modules.evaluate.src.config.namespaces import (
    compose_canonical,
    reject_conflicting_settings,
)
from modules.evaluate.src.config.suites import (
    EvaluationJob,
    ModelRef,
    SuiteError,
    expand_suite,
    resolve_model,
    resolve_model_collection,
    select_jobs,
)
from modules.evaluate.src.dataset.local_data import DataRootError, data_root_overrides
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.evaluation.preflight import (
    pin_dataset_views,
    pin_evaluation_config,
)
from modules.evaluate.src.evaluation.runner import run_evaluation
from modules.evaluate.src.tasks.engines import resolve_engine
from modules.evaluate.src.utils.baselines import SUPPORTED_BASELINE_TASKS
from modules.evaluate.src.utils.config_loader import load_evaluation_config
from modules.evaluate.src.utils.seed import seed_everything

_BASELINE_CAPABLE_SUITES = tuple(sorted(SUPPORTED_BASELINE_TASKS))

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
    parser.add_argument(
        "--data-root",
        help=(
            "Local dataset root: each dataset maps to DIR/<local_path or name>; "
            "missing inputs are errors (no Hub fallback)."
        ),
    )
    parser.add_argument(
        "--output-root",
        help="Benchmark output root (runtime.output_root); overrides the config file.",
    )
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
    data_root = Path(args.data_root) if args.data_root is not None else None
    if data_root is not None and not data_root.is_dir():
        raise SystemExit(f"--data-root is not a directory: {data_root}")
    shared: list[str] = []
    if args.seed is not None:
        shared.append(f"runtime.seed={args.seed}")
    if args.output_root is not None:
        shared.append(f"runtime.output_root={args.output_root}")
    if data_root is not None:
        shared.append(f"runtime.data_root={data_root}")
    if args.model_name is not None:
        shared.append(f"model.name={args.model_name}")
    try:
        reject_conflicting_settings(shared + list(args.overrides or []))
    except ValueError as exc:
        raise SystemExit(f"invalid overrides: {exc}") from exc
    models: list[ModelRef | None]
    if args.baselines:
        models = [None]
        # Baselines are all-or-nothing per request: a suite whose tasks have no
        # implemented shared baseline is an error, never a silent omission.
        unsupported = sorted({job.suite for job in jobs if not job.shared_baselines})
        if unsupported:
            raise SystemExit(
                "--baselines: no shared baseline is implemented for suite(s) "
                f"{', '.join(unsupported)}. Select only baseline-capable suites "
                f"({', '.join(sorted(_BASELINE_CAPABLE_SUITES))}) with --suite, "
                "without --baselines for model runs."
            )
        # Predictors do not apply, so jobs are unique per (task, dataset).
        seen: set[tuple[str, str]] = set()
        unique: list[EvaluationJob] = []
        for job in jobs:
            key = (job.task, job.dataset)
            if key in seen:
                continue
            seen.add(key)
            unique.append(_baseline_job(job))
        jobs = unique
    elif args.model_collection is not None:
        models = list(resolve_model_collection(args.model_collection))
    else:
        models = [resolve_model(args.model, label=args.model_name)]

    data_overrides: dict[tuple[str, str], list[str]] = {}
    runs: list[PlannedRun] = []
    for model in models:
        for job in jobs:
            config_paths = tuple(
                job.config_paths(model.config_path if model else None)
            ) + tuple(extra_files)
            base = (model.overrides if model is not None else ()) + tuple(job.overrides)
            # runtime.data_root comes from the composed config (YAML or --set) and
            # --data-root, which is already in ``shared``, wins over the files.
            root = _run_data_root(
                config_paths, base + tuple(shared) + tuple(args.overrides)
            )
            mapped: tuple[str, ...] = ()
            if root is not None:
                key = (job.dataset, str(root))
                if key not in data_overrides:
                    data_overrides[key] = data_root_overrides(
                        job.dataset_config_path(), root
                    )
                mapped = tuple(data_overrides[key])
            runs.append(
                PlannedRun(
                    job=job,
                    model_label=model.label if model is not None else None,
                    config_paths=config_paths,
                    overrides=base + mapped + tuple(shared) + tuple(args.overrides),
                )
            )
    return runs


def _run_data_root(paths: tuple[Path, ...], overrides: tuple[str, ...]) -> Path | None:
    """Return the effective ``runtime.data_root`` for one run, validated offline."""
    try:
        value = compose_canonical(list(paths), list(overrides)).get("data_root")
    except ValueError as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc
    if value is None:
        return None
    root = Path(str(value))
    if not root.is_dir():
        raise SystemExit(f"runtime.data_root is not a directory: {root}")
    return root


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
        f"seed={composed.seed} output_root={composed.workspace.base_path} "
        f"output_dir={composed.workspace.output_dir}"
    )


def _baseline_job(job: EvaluationJob) -> EvaluationJob:
    """Baseline-only job: no predictor, and baseline names for structural tasks.

    Sequence tasks declare ``baselines.names`` in YAML; structural tasks cannot
    (model runs reject it), so their default names are injected here instead.
    """
    from modules.evaluate.src.utils.baselines import STRUCTURAL_TASK_BASELINES

    overrides = job.overrides
    if job.task in STRUCTURAL_TASK_BASELINES:
        names = ",".join(job.shared_baselines)
        overrides = overrides + (f"baselines.names=[{names}]",)
    return replace(job, predictor=None, overrides=overrides)


def _run_baselines(run: PlannedRun) -> None:
    from modules.evaluate.src.runners.baselines import (
        load_baseline_config,
        run_baselines,
    )

    baseline_config = load_baseline_config(
        [str(path) for path in run.config_paths], list(run.overrides) or None
    )
    workspace, externals = pin_dataset_views(
        baseline_config.workspace, baseline_config.external_test_sets
    )
    baseline_config = baseline_config.model_copy(
        update={"workspace": workspace, "external_test_sets": externals}
    )
    for test, artifacts in run_baselines(baseline_config).items():
        for name, path in artifacts.items():
            logger.info("Baseline %s / %s: %s", test, name, path)


def _run_model_evaluation(
    config_paths: list[str], overrides: list[str], overwrite: bool
) -> None:
    """Run one model/dataset/predictor job end to end (legacy single-run body)."""
    logger.info("=== Starting Evaluation Pipeline ===")

    config = load_evaluation_config(path=config_paths, overrides=overrides or None)
    config = pin_evaluation_config(config)
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
    engine = resolve_engine(config)
    if engine is None:
        raise ValueError(
            f"No evaluation engine claims task_type='{config.workspace.task_type}' "
            f"with steps.mode='{config.steps.mode}'."
        )
    run_evaluation(config, workspace, accelerator, task=engine, overwrite=overwrite)

    if wandb_run is not None:
        wandb_run.finish()


def main(argv: list[str] | None = None) -> None:
    """Entry point: run a suite for one model, a collection, or baselines.

    Examples::

        uv run -m modules.evaluate.src.run_evaluate --suite calibration --model amplify_120m --dry-run
        uv run -m modules.evaluate.src.run_evaluate --suite contact_prediction --model esm2_35m --predictor contact_embedding
        uv run -m modules.evaluate.src.run_evaluate --suite sequence_regression --baselines
    """
    args = build_parser().parse_intermixed_args(sys.argv[1:] if argv is None else argv)
    _validate_args(args)
    try:
        runs = plan_runs(args)
    except (SuiteError, DataRootError) as error:
        raise SystemExit(f"error: {error}") from error

    if args.dry_run:
        for run in runs:
            summary = _compose_for_dry_run(run, baselines=args.baselines)
            print(f"[dry-run] {run.job.evaluation_id} | {summary}")
        print(f"[dry-run] {len(runs)} run(s) validated; nothing executed.")
        return

    if args.baselines:
        for run in runs:
            _run_baselines(run)
        return

    for run in runs:
        logger.info("Running %s model=%s", run.job.evaluation_id, run.model_label)
        _run_model_evaluation(
            [str(path) for path in run.config_paths],
            list(run.overrides),
            args.overwrite,
        )


if __name__ == "__main__":
    main()
