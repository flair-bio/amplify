"""Evaluation output directories remain independent of model loading paths."""

from pathlib import Path

import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspace,
    EvaluationWorkspaceConfig,
)


def workspace_config(base_dir, **overrides):
    return EvaluationWorkspaceConfig(
        **{
            "dataset_name": "toy_dataset",
            "dataset_repo_id": "flair-bio/toy-dataset",
            "model_name": "toy_model",
            "model_repo_id": "flair-bio/amplify-120m",
            "task_type": "sequence_classification",
            "target_type": "categorical",
            "label_format": "scalar",
            "num_labels": 2,
            "base_path": str(base_dir / "default"),
            **overrides,
        }
    )


@pytest.mark.parametrize(
    ("model_repo_id", "model_folder"),
    [
        (None, "toy_model"),
        ("flair-bio/amplify-120m", "flair-bio_amplify-120m"),
        ("/checkpoints/checkpoint_100000", "_checkpoints_checkpoint_100000"),
    ],
)
def test_default_output_layout_is_unchanged(tmp_path, model_repo_id, model_folder):
    config = workspace_config(tmp_path, model_repo_id=model_repo_id)
    workspace = EvaluationWorkspace(config)

    assert config.output_dir is None
    root = tmp_path / "default" / model_folder / "flair-bio_toy-dataset"
    assert workspace.scores_path == root / "scores"
    assert workspace.embeddings_path == root / "tmp/embeddings"


@pytest.mark.parametrize("variant", [None, "mlp-frozen_trunk"])
def test_output_dir_overrides_model_folder_without_changing_checkpoint(
    tmp_path, variant
):
    checkpoint = tmp_path / "checkpoints/checkpoint_100000"
    checkpoint.mkdir(parents=True)
    output = tmp_path / "custom results/01_baseline_checkpoint_100000"
    config = workspace_config(
        tmp_path, model_repo_id=str(checkpoint), output_dir=str(output)
    )
    workspace = EvaluationWorkspace(config, variant=variant)
    workspace.setup_directories()

    assert workspace.model_repo_id == str(checkpoint)
    assert workspace.model_name == "toy_model"
    assert config.output_dir == output
    root = output / "flair-bio_toy-dataset"
    run = root / "runs" / variant if variant else root
    assert workspace.model_path == run / "model"
    assert workspace.preds_path == run / "preds"
    assert workspace.stats_path == run / "stats"
    assert workspace.scores_path == run / "scores"
    assert workspace.run_manifest_path == run / "run_manifest.json"
    assert workspace.comparison_path == root / "scores"
    assert workspace.embeddings_path == root / "tmp/embeddings"
    assert workspace.model_path.is_dir()
    assert workspace.embeddings_path.is_dir()
    assert not (tmp_path / "default").exists()
    assert not (checkpoint.parent / "01_baseline_checkpoint_100000").exists()


def test_output_dir_preserves_an_explicit_temporary_directory(tmp_path):
    config = workspace_config(
        tmp_path,
        output_dir=tmp_path / "results",
        tmp_path=str(tmp_path / "cache"),
        dataset_repo_id=None,
    )
    workspace = EvaluationWorkspace(config)

    assert workspace.scores_path == tmp_path / "results/toy_dataset/scores"
    assert workspace.tmp_path == tmp_path / "cache"
    assert workspace.embeddings_path == tmp_path / "cache/embeddings"


def test_output_dir_yaml_override_parses_independently_of_model_path(tmp_path):
    configs = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"
    checkpoint = str(tmp_path / "checkpoints/checkpoint_100000")
    output = tmp_path / "01_baseline_checkpoint_100000"
    config = load_and_parse(
        [
            configs / "config.yaml",
            configs / "datasets/localization_prediction.yaml",
            configs / "tasks/sequence_classification.yaml",
        ],
        EvaluationConfig,
        overrides=[
            f"workspace.model_repo_id={checkpoint}",
            f"workspace.output_dir={output}",
        ],
    )

    assert config.workspace.model_repo_id == checkpoint
    assert config.workspace.output_dir == output
    assert config.model_dump(mode="json")["workspace"]["output_dir"] == str(output)
    workspace = EvaluationWorkspace(config.workspace, variant=config.variant)
    assert workspace.scores_path.is_relative_to(output)
