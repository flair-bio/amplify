"""The single command-line entry point for post-training workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from modules.core.utils.config_loader import load_and_parse
from modules.post_train.src.config import PostTrainingConfig

logger = logging.getLogger(__name__)


def parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-train an AMPLIFY model with latent structural alignment."
    )
    parser.add_argument(
        "config",
        nargs="*",
        help="YAML files followed by optional key=value overrides.",
    )
    args = parser.parse_args(argv)
    args.paths = [value for value in args.config if "=" not in value]
    args.overrides = [value for value in args.config if "=" in value]
    if not args.paths:
        parser.error("at least one YAML config file must be provided")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    config = load_and_parse(
        path=args.paths,
        model_cls=PostTrainingConfig,
        overrides=args.overrides or None,
    )
    if config.method == "latent_alignment":
        from modules.post_train.src.runners.latent_alignment import (
            run as run_latent_alignment,
        )

        run_latent_alignment(config)
    else:
        raise ValueError(f"Unsupported post-training method: {config.method!r}")
    logger.info("Post-training completed successfully.")


if __name__ == "__main__":
    main()
