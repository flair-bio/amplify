"""Disk-only standalone rescoring and explicitly selected paired comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.evaluate.src.artifacts.predictions import (
    load_predict_output,
    output_workspace,
)
from modules.evaluate.src.config.score import ScoreConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.scoring.run import compare_predictions, score_predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Saved model run_manifest.json."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Existing shared baseline name; repeatable.",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        help="Saved model control name; repeatable.",
    )
    parser.add_argument(
        "--score-config",
        type=Path,
        help="ScoreConfig JSON; defaults to the saved score/evaluation config.",
    )
    parser.add_argument(
        "--recover-context",
        action="store_true",
        help="Reconstruct missing provenance/crop coordinates from the saved evaluation config and original data (no model inference).",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    predict = manifest.get("predict", {})
    evaluation = manifest.get("evaluation", {}).get("config", {})
    workspace_config = predict.get("workspace") or evaluation.get("workspace")
    if not workspace_config:
        parser.error(
            "Manifest lacks workspace configuration; recover the original evaluation configuration first."
        )
    workspace = output_workspace(
        get_evaluation_workspace(
            EvaluationWorkspaceConfig.model_validate(workspace_config)
        ),
        manifest_path.parent,
    )
    raw_score = (
        json.loads(args.score_config.read_text())
        if args.score_config
        else manifest.get("score", {}).get("config")
        or evaluation.get("steps", {}).get("score", {})
    )
    score = ScoreConfig.model_validate(raw_score)
    output = load_predict_output(workspace)
    seed = predict.get("seed", 710019)
    if args.recover_context:
        from modules.evaluate.src.artifacts.recovery import recover_prediction_context

        output = recover_prediction_context(output, evaluation)
    if args.baseline or args.control:
        result = compare_predictions(
            score,
            workspace,
            baselines=args.baseline,
            controls=args.control,
            seed=seed,
            predict_result=output,
        )
    else:
        result = score_predictions(score, workspace, seed=seed, predict_result=output)
    print(result.scores_path)


if __name__ == "__main__":
    main()
