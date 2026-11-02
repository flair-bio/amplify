"""Dedicated ProteinGym runners share the ordinary workspace lifecycle."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
from torch import nn

from modules.core.utils.config_loader import load_runtime_config_from_argv
from modules.evaluate.src.proteingym import workspace as workspace_module
from modules.evaluate.src.proteingym.config import (
    ProteinGymSupervisedConfig,
    ProteinGymZeroShotConfig,
)

CONFIGS = Path(__file__).resolve().parents[2] / "modules/evaluate/configs"


@pytest.fixture(params=["zero_shot", "supervised"])
def run_case(request, monkeypatch, tmp_path):
    mode = request.param
    entrypoint = importlib.import_module(f"modules.evaluate.src.run_proteingym_{mode}")
    config_type = (
        ProteinGymZeroShotConfig if mode == "zero_shot" else ProteinGymSupervisedConfig
    )
    suite = tmp_path / "suite"
    data = suite / "proteingym-dms_substitutions"
    (data / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "dms_id": ["assay"],
            "uniprot_id": ["P1"],
            "target_seq": ["AC"],
            "coarse_selection_type": ["Activity"],
            "is_indel": [False],
            "is_clinical": [False],
        }
    ).write_parquet(data / "reference.parquet")
    pl.DataFrame(
        {
            "mutated_sequence": ["AC"] * 10,
            "mutant": ["A1C"] * 10,
            "dms_score": np.arange(10, dtype=float),
            "fold_random_5": np.arange(10) % 5,
        }
    ).write_parquet(data / "data/assay.parquet")
    load_model = Mock(return_value=nn.Identity())
    model_type = (
        entrypoint.AutoModelForMaskedLM if mode == "zero_shot" else entrypoint.AutoModel
    )
    monkeypatch.setattr(model_type, "from_pretrained", load_model)
    monkeypatch.setattr(
        entrypoint.AutoTokenizer, "from_pretrained", lambda *a, **k: object()
    )
    if mode == "zero_shot":
        monkeypatch.setattr(
            entrypoint,
            "_score_assay_model",
            lambda *a, **k: np.arange(10, dtype=float),
        )
    else:
        monkeypatch.setattr(
            entrypoint,
            "pool_embeddings",
            lambda *a, **k: np.arange(10, dtype=float).reshape(-1, 1),
        )
    args = [
        str(CONFIGS / f"proteingym_{mode}.yaml"),
        f"workspace.dataset_repo_id={suite}",
        f"workspace.output_dir={tmp_path / 'out'}",
        "workspace.model_name=amplify_checkpoint_100",
        "workspace.model_repo_id=/checkpoints/checkpoint_100",
        "workspace.run_id=test",
        "seed=42",
        "device=cpu",
    ]
    config = load_runtime_config_from_argv(args, config_type)
    variant = (
        "zero_shot" if mode == "zero_shot" else "linear-frozen_trunk-fold_random_5"
    )
    expected = (
        tmp_path
        / "out/proteingym/proteingym-dms_substitutions"
        / "amplify_checkpoint_100"
        / f"{variant}-test/seed_42"
    )
    return SimpleNamespace(
        entrypoint=entrypoint,
        config_type=config_type,
        config=config,
        args=args,
        expected=expected,
        load_model=load_model,
        data=data,
    )


def test_cli_resolves_shared_paths_and_protects_outputs(run_case, monkeypatch):
    case = run_case
    monkeypatch.setattr(sys, "argv", ["proteingym", *case.args])
    case.entrypoint.main()
    assert case.config.output_dir == case.expected
    assert not (case.expected / "summary.json").exists()
    assert not (case.expected / "assay_scores.parquet").exists()
    assert {p.name for p in case.expected.iterdir()} == {
        "predictions.parquet", "scores.parquet", "run_manifest.json"
    }
    predictions = pl.read_parquet(case.expected / "predictions.parquet")
    assert predictions.height == 10
    assert predictions["dms_id"].unique().to_list() == ["assay"]
    assert np.isfinite(predictions["prediction"].to_numpy()).all()
    if case.config_type is ProteinGymZeroShotConfig:
        np.testing.assert_array_equal(predictions["prediction"], np.arange(10))
    np.testing.assert_array_equal(predictions["label"], np.arange(10))
    assert predictions["id"].n_unique() == 10
    scores = pl.read_parquet(case.expected / "scores.parquet")
    assert scores.filter(
        (pl.col("scope") == "overall") & (pl.col("metric") == "spearman")
    )["value"].item() == pytest.approx(1.0)
    manifest = json.loads((case.expected / "run_manifest.json").read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["evaluation"]["config"]["workspace"]["model_name"] == (
        "amplify_checkpoint_100"
    )
    assert manifest["inputs"]["data_dir"] == str(case.data)
    assert manifest["workspace"]["model_name"] == "amplify_checkpoint_100"
    assert manifest["experiment"]["inputs"]["data_dir"] == str(case.data)
    assert manifest["score"]["scores"]["overall"] == pytest.approx(1.0)
    assert manifest["score"]["scores_path"] == str(case.expected / "scores.parquet")
    assert "assays" not in manifest["score"]["condition_scores"]["primary"]
    assert not list(case.expected.parent.glob("experiment_*.json"))
    assert not (case.expected.parent.parent / "workspace_manifest.json").exists()
    case.load_model.assert_called_once()

    with pytest.raises(FileExistsError, match="already exist"):
        case.entrypoint.main()
    case.load_model.assert_called_once()
    marker = case.expected / "obsolete"
    marker.touch()
    monkeypatch.setattr(sys, "argv", ["proteingym", "--overwrite", *case.args])
    case.entrypoint.main()
    assert not marker.exists()
    assert case.load_model.call_count == 2


def test_sibling_seeds_share_experiment_but_not_outputs(run_case):
    case = run_case
    case.entrypoint.run(case.config)
    other = load_runtime_config_from_argv([*case.args, "seed=43"], case.config_type)
    case.entrypoint.run(other)
    assert other.output_dir != case.config.output_dir
    assert other.output_dir.name == "seed_43"
    assert (case.expected / "run_manifest.json").is_file()
    assert (other.output_dir / "run_manifest.json").is_file()


def test_workspace_records_failure_and_releases_run_lock(run_case):
    case = run_case
    case.load_model.side_effect = RuntimeError("model failed")
    with pytest.raises(RuntimeError, match="model failed"):
        case.entrypoint.run(case.config)
    manifest = json.loads((case.expected / "run_manifest.json").read_text())
    assert manifest["evaluation"]["status"] == "failed"
    assert manifest["evaluation"]["error"] == "model failed"
    case.load_model.side_effect = None
    case.entrypoint.run(case.config, overwrite=True)
    manifest = json.loads((case.expected / "run_manifest.json").read_text())
    assert manifest["evaluation"]["status"] == "complete"


def test_workspace_blocks_concurrent_model_work(run_case):
    with (
        workspace_module.protein_gym_run(run_case.config),
        pytest.raises(FileExistsError, match="already running"),
    ):
        run_case.entrypoint.run(run_case.config)
    run_case.load_model.assert_not_called()


def test_hub_suite_download_is_scoped_and_revision_pinned(run_case, monkeypatch):
    case = run_case
    snapshot = Mock(return_value=str(case.data.parent))
    monkeypatch.setattr(workspace_module, "snapshot_download", snapshot)
    config = load_runtime_config_from_argv(
        [
            *case.args,
            "workspace.dataset_repo_id=flair-bio/evaluation-suite",
            f"workspace.dataset_revision={'a' * 40}",
            "force_download=true",
        ],
        case.config_type,
    )
    case.entrypoint.run(config)
    snapshot.assert_called_once_with(
        repo_id="flair-bio/evaluation-suite",
        repo_type="dataset",
        revision="a" * 40,
        allow_patterns=["proteingym-dms_substitutions/**"],
        force_download=True,
    )
    assert (case.expected / "run_manifest.json").is_file()


def test_missing_suite_is_rejected_before_model_work(run_case):
    case = run_case
    config = load_runtime_config_from_argv(
        [
            *case.args,
            f"workspace.dataset_repo_id={case.data.parent / 'missing'}",
        ],
        case.config_type,
    )
    with pytest.raises(FileNotFoundError, match="suite not found"):
        case.entrypoint.run(config)
    case.load_model.assert_not_called()


def test_suite_subfolder_can_differ_from_output_label(run_case):
    case = run_case
    config = load_runtime_config_from_argv(
        [
            *case.args,
            "workspace.dataset_name=custom_label",
        ],
        case.config_type,
    )
    case.entrypoint.run(config)
    assert "custom_label" in config.output_dir.parts
    assert (config.output_dir / "run_manifest.json").is_file()


def test_scratch_root_locks_and_overwrite_preserve_siblings(run_case, tmp_path):
    case = run_case
    config = load_runtime_config_from_argv(
        [*case.args, f"workspace.tmp_path={tmp_path / 'scratch'}"],
        case.config_type,
    )
    case.entrypoint.run(config)
    sibling = load_runtime_config_from_argv(
        [*case.args, f"workspace.tmp_path={tmp_path / 'scratch'}", "seed=43"],
        case.config_type,
    )
    case.entrypoint.run(sibling)
    sibling_manifest = (sibling.output_dir / "run_manifest.json").read_bytes()
    case.entrypoint.run(config, overwrite=True)
    assert (sibling.output_dir / "run_manifest.json").read_bytes() == sibling_manifest
    workspace = workspace_module.get_evaluation_workspace(
        config.workspace, variant=config.run_variant(config.model_dump()), seed=config.seed
    )
    expected = tmp_path / "scratch/proteingym/proteingym-dms_substitutions/amplify_checkpoint_100"
    assert workspace.embeddings_path == expected / "embeddings"
    assert workspace.locks_path.parent == expected / "locks"
    assert list(workspace.locks_path.glob("*.lock"))
    assert not list(config.output_dir.parent.parent.glob("*.lock"))


def test_zero_shot_conditions_publish_in_owning_seed_manifest(run_case):
    case = run_case
    if case.config_type is not ProteinGymZeroShotConfig:
        return
    config = load_runtime_config_from_argv(
        [*case.args, "baselines=[random,blosum62]"], case.config_type
    )
    case.entrypoint.run(config)
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
    assert set(manifest["score"]["condition_scores"]) == {"primary", "random", "blosum62"}
    assert {p.name for p in config.output_dir.iterdir()} == {
        "predictions.parquet", "scores.parquet", "run_manifest.json"
    }
    predictions = pl.read_parquet(config.output_dir / "predictions.parquet")
    scores = pl.read_parquet(config.output_dir / "scores.parquet")
    assert predictions.height == 30
    for name in ("primary", "random", "blosum62"):
        assert predictions.filter(pl.col("control") == name).height == 10
        assert scores.filter(
            (pl.col("control") == name) & (pl.col("scope") == "assay")
            & (pl.col("metric") == "spearman")
        ).height == 1
    case.entrypoint.run(config, overwrite=True)
    manifest = json.loads((config.output_dir / "run_manifest.json").read_text())
    assert set(manifest["score"]["condition_scores"]) == {"primary", "random", "blosum62"}
    assert pl.read_parquet(config.output_dir / "predictions.parquet").height == 30


def test_combined_predictions_preserve_assay_identity_and_optional_columns(run_case):
    case = run_case
    reference = pl.read_parquet(case.data / "reference.parquet")
    second = reference.with_columns(
        pl.lit("other").alias("dms_id"), pl.lit("P2").alias("uniprot_id")
    )
    pl.concat([reference, second]).write_parquet(case.data / "reference.parquet")
    assay = pl.read_parquet(case.data / "data/assay.parquet").with_columns(
        pl.Series("id", [f"source-{i}" for i in range(10)]),
        pl.lit("retained").alias("extra"),
    )
    assay.write_parquet(case.data / "data/other.parquet")
    case.entrypoint.run(case.config)
    predictions = pl.read_parquet(case.expected / "predictions.parquet")
    assert predictions.height == 20
    assert predictions.filter(pl.col("dms_id") == "other")["id"].to_list() == assay["id"].to_list()
    assert predictions.filter(pl.col("dms_id") == "assay")["extra"].null_count() == 10
    scores = pl.read_parquet(case.expected / "scores.parquet")
    spearman = scores.filter(pl.col("metric") == "spearman")
    assert spearman.filter(pl.col("scope") == "assay").height == 2
    assert spearman.filter(pl.col("scope") == "function")["value"].item() == pytest.approx(1.0)
    manifest = json.loads((case.expected / "run_manifest.json").read_text())
    assert spearman.filter(pl.col("scope") == "overall")["value"].item() == pytest.approx(
        manifest["score"]["scores"]["overall"]
    )


def test_supervised_scores_preserve_official_scheme_metric_override(run_case, monkeypatch):
    case = run_case
    if case.config_type is not ProteinGymSupervisedConfig:
        return
    assay_path = case.data / "data/assay.parquet"
    assay = pl.read_parquet(assay_path).with_columns(
        pl.col("fold_random_5").alias("fold_modulo_5"),
        pl.col("fold_random_5").alias("fold_contiguous_5"),
    )
    assay.write_parquet(assay_path)
    monkeypatch.setattr(case.entrypoint, "run_cv", lambda *a, **k: np.arange(10, dtype=float))
    monkeypatch.setattr(case.entrypoint, "compute_assay_metric", lambda *a, **k: ("spearman", 0.25))
    config = load_runtime_config_from_argv(
        [*case.args, "split=official_average"], case.config_type
    )
    case.entrypoint.run(config)
    scores = pl.read_parquet(config.output_dir / "scores.parquet")
    assert scores["value"].to_list() == pytest.approx([0.25, 0.25, 0.25])
    predictions = pl.read_parquet(config.output_dir / "predictions.parquet")
    np.testing.assert_array_equal(predictions["prediction"], np.arange(10))


def test_multiple_processes_are_rejected_before_model_work(run_case, monkeypatch):
    monkeypatch.setattr(
        workspace_module, "PartialState", lambda: SimpleNamespace(num_processes=2)
    )
    with pytest.raises(ValueError, match="--num_processes 1"):
        run_case.entrypoint.run(run_case.config)
    run_case.load_model.assert_not_called()


@pytest.mark.parametrize(
    "override,error",
    [
        ("workspace.model_name=../escape", "Output labels"),
        ("workspace.dataset_config_name=../escape", "Output labels"),
        ("workspace.task_type=sequence_regression", "must be 'proteingym'"),
        ("workspace.dataset_revision=main", "must be null for local"),
        ("model_id=another_model", "Do not mix"),
        ("data_repo_id=org/data", "Use workspace.dataset_repo_id"),
    ],
)
def test_invalid_workspace_settings_fail_early(run_case, override, error):
    with pytest.raises(ValueError, match=error):
        load_runtime_config_from_argv([*run_case.args, override], run_case.config_type)
