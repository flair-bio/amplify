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
        / "out/proteingym/proteingym-dms_substitutions/models"
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
    assert (case.expected / "summary.json").is_file()
    assert (case.expected / "assay_scores.parquet").is_file()
    assert (case.expected / "assay_scored.parquet").is_file()
    manifest = json.loads((case.expected / "run_manifest.json").read_text())
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["evaluation"]["config"]["workspace"]["model_name"] == (
        "amplify_checkpoint_100"
    )
    assert manifest["inputs"]["data_dir"] == str(case.data)
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
    assert (case.expected / "summary.json").is_file()
    assert (other.output_dir / "summary.json").is_file()


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
    assert (case.expected / "summary.json").is_file()


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
    assert (config.output_dir / "summary.json").is_file()


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
