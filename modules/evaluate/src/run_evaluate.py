"""The single command-line entry point for all evaluation workflows."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from typing import Any

from modules.evaluate.src.config.evaluation import BaselineRunConfig
from modules.evaluate.src.config.load import RunConfig, load_run_config
from modules.evaluate.src.runners.runtime import Runtime, initialize_runtime

logger = logging.getLogger(__name__)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one configured target or a ProteinGym assay suite."
    )
    parser.add_argument(
        "--baselines", action="store_true", help="Run ordinary dataset baselines only."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing model run."
    )
    parser.add_argument(
        "config", nargs="*", help="YAML files followed by optional key=value overrides."
    )
    args = parser.parse_args(argv)
    if args.baselines and args.overwrite:
        parser.error(
            "--overwrite applies to model runs only; baseline artifacts are validated and reused."
        )
    args.paths = [value for value in args.config if "=" not in value]
    args.overrides = [value for value in args.config if "=" in value]
    if not args.paths:
        parser.error("at least one YAML config file must be provided")
    return args


def dispatch_run(
    config: RunConfig,
    runtime: Runtime,
    *,
    overwrite: bool = False,
) -> Any:
    if isinstance(config, BaselineRunConfig):
        from modules.evaluate.src.runners.baselines import run_baselines

        return run_baselines(config)
    if config.workflow == "proteingym_supervised":
        from modules.evaluate.src.proteingym.supervised import run_supervised

        return run_supervised(config, runtime, overwrite=overwrite)
    if config.workflow == "proteingym_zero_shot":
        from modules.evaluate.src.proteingym.zero_shot import run_zero_shot

        return run_zero_shot(config, runtime, overwrite=overwrite)
    from modules.evaluate.src.runners.evaluate import run_evaluation

    return run_evaluation(config, runtime, overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli(argv)
    config = load_run_config(
        args.paths, args.overrides or None, baseline_only=args.baselines
    )
    with initialize_runtime(config) as runtime:
        dispatch_run(config, runtime, overwrite=args.overwrite)
    logger.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
