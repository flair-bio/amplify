"""Evaluation output directories remain independent of model loading paths."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import polars as pl
import pytest

from modules.core.utils.config_loader import load_and_parse
from modules.evaluate.src.contracts.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.contracts.config import EvaluationWorkspaceConfig


def workspace_config(base_dir, **overrides):
    return EvaluationWorkspaceConfig(
        **{
            "dataset_name": "toy_dataset",
            "run_id": "test",
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
    "model_repo_id",
    [
        None,
        "flair-bio/amplify-120m",
        "/checkpoints/checkpoint_100000",
    ],
)
def test_output_layout_uses_labels_not_source_paths(tmp_path, model_repo_id):
    config = workspace_config(tmp_path, model_repo_id=model_repo_id)
    workspace = EvaluationWorkspace(config)

    assert config.output_dir is None
    root = tmp_path / "default/sequence_classification/toy_dataset/models/toy_model"
    assert workspace.scores_path == root / "test"
    assert workspace.embeddings_path == root / "tmp/embeddings"


@pytest.mark.parametrize("variant", [None, "mlp-frozen_trunk"])
def test_output_dir_overrides_experiment_root_without_changing_checkpoint(
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
    root = output / "sequence_classification/toy_dataset/models/toy_model"
    run = root / (f"{variant}-test" if variant else "test")
    assert workspace.model_path == run / "model"
    assert workspace.preds_path == run
    assert workspace.stats_path == run
    assert workspace.scores_path == run
    assert workspace.run_manifest_path == run / "run_manifest.json"
    assert workspace.comparison_path == run
    assert workspace.embeddings_path == root / "tmp/embeddings"
    assert workspace.run_root.is_dir()
    assert not workspace.model_path.exists()
    assert not workspace.embeddings_path.exists()
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

    assert workspace.scores_path == (
        tmp_path / "results/sequence_classification/toy_dataset/models/toy_model/test"
    )
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


@pytest.mark.parametrize("field", ["run_id", "dataset_name", "model_name"])
@pytest.mark.parametrize("value", ["", ".", "..", "../escape", "/absolute", "a/b"])
def test_output_labels_cannot_escape_directories(tmp_path, field, value):
    with pytest.raises(ValueError, match="Output labels"):
        workspace_config(tmp_path, **{field: value})


def test_run_id_is_required(tmp_path):
    payload = workspace_config(tmp_path).model_dump()
    del payload["run_id"]
    with pytest.raises(ValueError, match="run_id"):
        EvaluationWorkspaceConfig.model_validate(payload)


@pytest.mark.parametrize(
    "change", [{"dataset_config_name": "other"}, {"dataset_revision": "b" * 40}]
)
def test_suite_configs_and_revisions_cannot_reuse_output_labels(tmp_path, change):
    config = workspace_config(
        tmp_path, dataset_config_name="main", dataset_revision="a" * 40
    )
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    with workspace.reserve_run():
        pass
    changed = EvaluationWorkspace(
        config.model_copy(update=change), variant="mlp-frozen_trunk"
    )
    with pytest.raises(ValueError, match="different inputs"):
        with changed.reserve_run():
            pass


def test_checkpoint_versions_and_tasks_have_separate_roots(tmp_path):
    first = EvaluationWorkspace(workspace_config(tmp_path), variant="mlp-frozen_trunk")
    second = EvaluationWorkspace(
        workspace_config(
            tmp_path, model_name="amplify-350m-muon", model_repo_id="/checkpoints/muon"
        ),
        variant="mlp-frozen_trunk",
    )
    third = EvaluationWorkspace(
        workspace_config(
            tmp_path,
            task_type="sequence_regression",
            target_type="continuous",
            num_labels=None,
        ),
        variant="mlp-frozen_trunk",
    )
    assert first.workspace_root != second.workspace_root != third.workspace_root
    assert first.workspace_root.parent == second.workspace_root.parent


def test_run_ids_share_cache_but_keep_results_and_overwrite_separate(tmp_path):
    config = workspace_config(tmp_path)
    first = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    second = EvaluationWorkspace(
        config.model_copy(update={"run_id": "repeat"}), variant="mlp-frozen_trunk"
    )
    with first.reserve_run() as targets:
        first.initialize_run(targets, overwrite=False, config={})
    before = first.run_manifest_path.read_bytes()
    with second.reserve_run() as targets:
        second.initialize_run(targets, overwrite=False, config={})
    with second.reserve_run(overwrite=True) as targets:
        second.initialize_run(targets, overwrite=True, config={})
    assert first.run_root != second.run_root
    assert first.embeddings_path == second.embeddings_path
    assert first.run_manifest_path.read_bytes() == before


def test_control_artifacts_live_inside_the_owning_run(tmp_path):
    workspace = EvaluationWorkspace(workspace_config(tmp_path), variant="mlp")
    control = workspace.control_workspace("random_trunk")
    assert control.run_root == workspace.run_root / "controls/random_trunk"
    assert control.model_path == control.run_root / "model"
    assert control.preds_path == control.run_root
    with pytest.raises(ValueError, match="Output labels"):
        workspace.control_workspace("../escape")


@pytest.mark.parametrize(
    "output", ["model/checkpoint.bin", "preds/data.parquet", "run_manifest.json"]
)
def test_existing_variant_requires_explicit_replacement(tmp_path, output):
    workspace = EvaluationWorkspace(
        workspace_config(tmp_path), variant="mlp-frozen_trunk"
    )
    path = workspace.run_root / output
    path.parent.mkdir(parents=True)
    path.write_text("existing data")
    with pytest.raises(FileExistsError, match="--overwrite"), workspace.reserve_run():
        pytest.fail("must reject before model work")
    assert path.read_text() == "existing data"


@pytest.mark.parametrize(
    "comparison_name", ["scores_long.parquet", "layerwise_comparison.parquet"]
)
def test_overwrite_cleans_only_target_and_derived_results(tmp_path, comparison_name):
    config = workspace_config(tmp_path)
    workspace = EvaluationWorkspace(
        config, variant="layerwise-torch_linear-frozen_trunk"
    )
    layer = EvaluationWorkspace(config, variant=f"{workspace.variant}-layer_2")
    random = workspace.control_workspace("random_trunk")
    sibling = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    unrelated = EvaluationWorkspace(config, variant=f"{workspace.variant}-other")
    for item in (workspace, layer, random, sibling, unrelated):
        item.setup_directories()
        item.model_path.mkdir()
        (item.model_path / "weights").write_text("checkpoint")
    embeddings = workspace.embeddings_path / "features"
    embeddings.parent.mkdir(parents=True)
    embeddings.write_text("cached embeddings")
    baseline = config.dataset_root / "baselines/train_mean/predictions.parquet"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("shared baseline")
    comparison = workspace.comparison_path / comparison_name
    pl.DataFrame(
        {
            "variant": [workspace.variant, layer.variant, sibling.variant],
            "value": [1, 2, 3],
        }
    ).write_parquet(comparison)
    sibling_comparison = sibling.comparison_path / comparison_name
    pl.DataFrame({"variant": [sibling.variant], "value": [3]}).write_parquet(
        sibling_comparison
    )
    sibling_scores = sibling_comparison.read_bytes()

    with workspace.reserve_run(overwrite=True) as targets:
        workspace.initialize_run(targets, overwrite=True, config={"seed": 1})

    assert not (workspace.model_path / "weights").exists()
    assert not layer.run_root.exists()
    assert not random.run_root.exists()
    assert (sibling.model_path / "weights").read_text() == "checkpoint"
    assert (unrelated.model_path / "weights").read_text() == "checkpoint"
    assert embeddings.read_text() == "cached embeddings"
    assert baseline.read_text() == "shared baseline"
    assert not comparison.exists()
    assert sibling_comparison.read_bytes() == sibling_scores
    assert json.loads(workspace.run_manifest_path.read_text())["evaluation"][
        "status"
    ] == ("running")


def test_interrupted_layer_outputs_are_protected_without_base_predictions(tmp_path):
    config = workspace_config(tmp_path)
    workspace = EvaluationWorkspace(config, variant="layerwise")
    layer = EvaluationWorkspace(config, variant="layerwise-layer_7")
    layer.setup_directories()
    layer.model_path.mkdir()
    (layer.model_path / "weights").write_text("partial")
    with pytest.raises(FileExistsError, match="layer_7"), workspace.reserve_run():
        pass


def test_active_run_cannot_be_overwritten(tmp_path):
    workspace = EvaluationWorkspace(
        workspace_config(tmp_path), variant="mlp-frozen_trunk"
    )
    with (
        workspace.reserve_run(),
        pytest.raises(FileExistsError, match="already running"),
        workspace.reserve_run(overwrite=True),
    ):
        pass


def test_model_labels_cannot_be_reused_for_different_sources(tmp_path):
    config = workspace_config(tmp_path)
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    with workspace.reserve_run():
        pass
    renamed_source = EvaluationWorkspace(
        config.model_copy(update={"model_repo_id": "/another/checkpoint"}),
        variant="torch_linear-frozen_trunk",
    )
    with (
        pytest.raises(ValueError, match="different inputs"),
        renamed_source.reserve_run(overwrite=True),
    ):
        pass


def test_overwrite_rejects_symlinked_outputs(tmp_path):
    workspace = EvaluationWorkspace(
        workspace_config(tmp_path), variant="mlp-frozen_trunk"
    )
    source = tmp_path / "original"
    source.mkdir()
    (source / "weights").write_text("keep")
    workspace.run_root.parent.mkdir(parents=True)
    workspace.run_root.symlink_to(source, target_is_directory=True)
    with (
        pytest.raises(ValueError, match="symlinked"),
        workspace.reserve_run(overwrite=True),
    ):
        pass
    assert (source / "weights").read_text() == "keep"


def test_overwrite_rejects_checkpoint_inside_target(tmp_path):
    config = workspace_config(tmp_path)
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    checkpoint = workspace.model_path / "source"
    checkpoint.mkdir(parents=True)
    (checkpoint / "weights").write_text("keep")
    config.model_repo_id = str(checkpoint)
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    with (
        pytest.raises(ValueError, match="overlaps input/cache"),
        workspace.reserve_run(overwrite=True),
    ):
        pass
    assert (checkpoint / "weights").read_text() == "keep"


def test_cli_passes_explicit_overwrite_to_pipeline(tmp_path, monkeypatch):
    from modules.evaluate.src import run_evaluate

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"workspace": workspace_config(tmp_path).model_dump(mode="json")})
    )
    monkeypatch.setattr("sys.argv", ["run_evaluate.py", "--overwrite", str(path)])
    monkeypatch.setattr(
        run_evaluate,
        "Accelerator",
        lambda **kwargs: SimpleNamespace(is_main_process=True),
    )
    monkeypatch.setattr(run_evaluate, "seed_everything", lambda seed: None)
    run = Mock()
    monkeypatch.setattr(run_evaluate, "run_pipeline", run)
    run_evaluate.main()
    assert run.call_args.kwargs["overwrite"] is True
    assert run.call_args.args[1].config.run_id == "test"


def test_cli_rejects_baseline_overwrite_without_touching_artifacts(monkeypatch):
    from modules.evaluate.src import run_evaluate

    monkeypatch.setattr("sys.argv", ["run_evaluate.py", "--baselines", "--overwrite"])
    with pytest.raises(SystemExit, match="model runs only"):
        run_evaluate.main()
