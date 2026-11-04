"""Property-isolated ProtParams runs and strict equal-weight summaries."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from modules.evaluate.src.contracts.config_loader import load_evaluation_config
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.metrics.tables import aggregate_protparams
from modules.evaluate.src.runners import protparams
from modules.evaluate.src.utils.io import update_run_manifest, write_json
from modules.evaluate.src.utils.protparams import PROTPARAM_COLUMNS

from .test_config_loader import CONFIGS


@pytest.fixture
def property_runs(tmp_path):
    root = tmp_path / "sequence_regression/uniprot_protparams/toy"
    config = load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/uniprot_protparams.yaml"],
        overrides=["workspace.model_name=toy", f"workspace.output_dir={tmp_path}"],
    )
    run_name = f"{config.variant}-{config.workspace.run_id}"
    paths = {}
    for index, prop in enumerate(PROTPARAM_COLUMNS):
        run = config.model_dump(mode="json")
        run["protparams_all_properties"] = False
        run["workspace"]["run_id"] += f"-{prop}"
        run["steps"]["prepare"]["label_column"] = prop
        path = root / f"{run_name}-{prop}/seed_{config.seed}/run_manifest.json"
        write_json(
            path,
            {
                "evaluation": {"status": "complete", "config": run},
                "score": {
                    "metadata": {
                        "model_name": "toy",
                        "dataset_name": "uniprot_protparams",
                        "task_type": "sequence_regression",
                        "split": "default",
                        "variant": config.variant,
                        "seed": config.seed,
                    },
                    "scores": {"r2": index - 20.0, "mse": index * 100.0},
                },
            },
        )
        paths[prop] = path
    return root, run_name, config.seed, paths


def test_properties_have_distinct_outputs_and_keep_dataset_identity(tmp_path):
    workspaces = []
    for prop in PROTPARAM_COLUMNS:
        config = load_evaluation_config(
            [
                CONFIGS / "config.yaml",
                CONFIGS / "tasks/sequence_regression.yaml",
                CONFIGS / "datasets/uniprot_protparams.yaml",
            ],
            overrides=[
                "protparams_all_properties=false",
                f"steps.prepare.label_column={prop}",
                f"workspace.run_id=run01-{prop}",
                f"workspace.output_dir={tmp_path}",
            ],
        )
        workspace = get_evaluation_workspace(
            config.workspace, variant=config.variant, seed=config.seed
        )
        assert workspace.config.dataset_name == "uniprot_protparams"
        assert workspace.config.dataset_config_name == "uniprot_protparams"
        assert config.steps.prepare.label_column == prop
        assert config.steps.tune.hyperparameter_search.metric == "r2"
        assert config.steps.score.metrics == ["r2", "mse"]
        workspaces.append(workspace)
    assert len({workspace.model_path for workspace in workspaces}) == len(PROTPARAM_COLUMNS)
    assert len({workspace.run_manifest_path for workspace in workspaces}) == len(PROTPARAM_COLUMNS)
    assert len({workspace.embeddings_path for workspace in workspaces}) == 1


def test_summary_averages_properties_equally_and_preserves_negative_r2(property_runs):
    root, run_name, seed, _ = property_runs
    table, summary = aggregate_protparams(root, run_name, seed)
    assert table["property"].to_list() == list(PROTPARAM_COLUMNS)
    assert table["r2"].to_list() == [index - 20.0 for index in range(len(PROTPARAM_COLUMNS))]
    assert summary["value"] == pytest.approx(-5.0)
    assert summary["n_properties"] == len(PROTPARAM_COLUMNS)
    assert summary["aggregation"] == "unweighted_property_mean"


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", FileNotFoundError),
        ("incomplete", ValueError),
        ("nan", ValueError),
        ("infinite", ValueError),
        ("wrong_property", ValueError),
        ("wrong_run_id", ValueError),
        ("wrong_seed", ValueError),
        ("different_split", ValueError),
        ("different_model", ValueError),
        ("different_protocol", ValueError),
        ("external_test", ValueError),
    ],
)
def test_summary_rejects_missing_invalid_or_mixed_runs(property_runs, change, error):
    root, run_name, seed, paths = property_runs
    path = paths["gravy"]
    if change == "missing":
        path.unlink()
    else:
        manifest = json.loads(path.read_text())
        config = manifest["evaluation"]["config"]
        metadata = manifest["score"]["metadata"]
        if change == "incomplete":
            manifest["evaluation"]["status"] = "running"
        elif change in {"nan", "infinite"}:
            manifest["score"]["scores"]["r2"] = float("nan" if change == "nan" else "inf")
        elif change == "wrong_property":
            config["steps"]["prepare"]["label_column"] = "mass"
        elif change == "wrong_run_id":
            config["workspace"]["run_id"] = "other"
        elif change == "wrong_seed":
            metadata["seed"] += 1
        elif change == "different_split":
            metadata["split"] = "stratified"
        elif change == "different_model":
            config["workspace"]["model_repo_id"] = "/another/checkpoint"
        elif change == "different_protocol":
            config["steps"]["prepare"]["max_length"] = 1024
        elif change == "external_test":
            metadata["test_subset"] = "other"
        write_json(path, manifest)
    with pytest.raises(error):
        aggregate_protparams(root, run_name, seed)


def _suite_config(tmp_path):
    return load_evaluation_config(
        [CONFIGS / "config.yaml", CONFIGS / "datasets/uniprot_protparams.yaml"],
        overrides=["workspace.model_name=toy", f"workspace.output_dir={tmp_path}"],
    )


@pytest.fixture
def suite_pipeline(monkeypatch):
    calls = []

    def run_pipeline(config, workspace, accelerator, *, overwrite):
        calls.append((config, workspace, overwrite))
        with workspace.reserve_run(overwrite=overwrite) as targets:
            workspace.initialize_run(
                targets, overwrite=overwrite, config=config.model_dump(mode="json")
            )
            prop = config.steps.prepare.label_column
            metadata = {
                "model_name": config.workspace.model_name,
                "dataset_name": config.workspace.dataset_name,
                "task_type": config.workspace.task_type,
                "split": config.steps.prepare.split.name,
                "variant": config.variant,
                "seed": config.seed,
            }
            update_run_manifest(
                workspace.run_manifest_path,
                "score",
                {"metadata": metadata, "scores": {"r2": PROTPARAM_COLUMNS.index(prop) - 20.0}},
            )
            update_run_manifest(
                workspace.run_manifest_path,
                "evaluation",
                {"status": "complete", "config": config.model_dump(mode="json")},
            )

    monkeypatch.setattr(protparams, "run_pipeline", run_pipeline)
    return calls


def _accelerator():
    return SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        wait_for_everyone=MagicMock(),
        free_memory=MagicMock(),
    )


def test_suite_runs_all_properties_publishes_and_resumes(tmp_path, suite_pipeline):
    config = _suite_config(tmp_path)
    accelerator = _accelerator()
    summary = protparams.run_protparams(config, accelerator)
    assert summary["value"] == pytest.approx(-5.0)
    assert [child.steps.prepare.label_column for child, _, _ in suite_pipeline] == list(PROTPARAM_COLUMNS)
    assert all(not child.protparams_all_properties for child, _, _ in suite_pipeline)
    assert config.protparams_all_properties
    assert config.steps.prepare.label_column == "gravy"
    assert config.workspace.run_id == "run01"
    assert accelerator.free_memory.call_count == len(PROTPARAM_COLUMNS)
    output = config.workspace.dataset_root / "toy" / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert pl.read_parquet(output / "property_scores.parquet").height == len(PROTPARAM_COLUMNS)
    assert json.loads((output / "summary.json").read_text())["value"] == pytest.approx(-5.0)
    assert json.loads((output / "run_manifest.json").read_text())["evaluation"]["status"] == "complete"
    assert protparams.run_protparams(config, accelerator) == summary
    assert len(suite_pipeline) == len(PROTPARAM_COLUMNS)


def test_suite_restarts_only_matching_incomplete_properties(tmp_path, suite_pipeline):
    config = _suite_config(tmp_path)
    protparams.run_protparams(config, _accelerator())
    _, workspace, _ = suite_pipeline[0]
    manifest = json.loads(workspace.run_manifest_path.read_text())
    manifest["evaluation"]["status"] = "running"
    write_json(workspace.run_manifest_path, manifest)
    protparams.run_protparams(config, _accelerator())
    assert len(suite_pipeline) == len(PROTPARAM_COLUMNS) + 1
    assert suite_pipeline[-1][0].steps.prepare.label_column == PROTPARAM_COLUMNS[0]
    assert suite_pipeline[-1][2]


def test_suite_refuses_incompatible_results_before_running_properties(tmp_path, suite_pipeline):
    config = _suite_config(tmp_path)
    protparams.run_protparams(config, _accelerator())
    _, workspace, _ = suite_pipeline[-1]
    manifest = json.loads(workspace.run_manifest_path.read_text())
    manifest["evaluation"]["config"]["steps"]["prepare"]["max_length"] += 1
    write_json(workspace.run_manifest_path, manifest)
    with pytest.raises(ValueError, match="incompatible settings"):
        protparams.run_protparams(config, _accelerator())
    assert len(suite_pipeline) == len(PROTPARAM_COLUMNS)


def test_suite_overwrite_reruns_every_property(tmp_path, suite_pipeline):
    config = _suite_config(tmp_path)
    protparams.run_protparams(config, _accelerator())
    protparams.run_protparams(config, _accelerator(), overwrite=True)
    assert len(suite_pipeline) == 2 * len(PROTPARAM_COLUMNS)
    assert all(overwrite for _, _, overwrite in suite_pipeline[len(PROTPARAM_COLUMNS):])


def test_suite_failure_does_not_publish_partial_or_stale_summary(tmp_path, suite_pipeline, monkeypatch):
    config = _suite_config(tmp_path)
    protparams.run_protparams(config, _accelerator())

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted property")

    monkeypatch.setattr(protparams, "run_pipeline", fail)
    with pytest.raises(RuntimeError, match="interrupted property"):
        protparams.run_protparams(config, _accelerator(), overwrite=True)
    output = config.workspace.dataset_root / "toy" / f"{config.variant}-run01-summary/seed_{config.seed}"
    assert not (output / "summary.json").exists()
    assert not (output / "property_scores.parquet").exists()
    assert json.loads((output / "run_manifest.json").read_text())["evaluation"]["status"] == "running"


@pytest.mark.parametrize("all_properties", [True, False])
def test_entry_point_dispatches_suite_or_single_property(
    tmp_path, monkeypatch, all_properties
):
    import sys
    from modules.evaluate.src import run_evaluate

    config = _suite_config(tmp_path)
    config.protparams_all_properties = all_properties
    suite, single = MagicMock(), MagicMock()
    monkeypatch.setattr(sys, "argv", ["run_evaluate", "config.yaml"])
    monkeypatch.setattr(run_evaluate, "load_evaluation_config", lambda **kwargs: config)
    monkeypatch.setattr(run_evaluate, "Accelerator", lambda **kwargs: _accelerator())
    monkeypatch.setattr(run_evaluate, "run_protparams", suite)
    monkeypatch.setattr(run_evaluate, "run_pipeline", single)
    run_evaluate.main()
    assert suite.call_count == int(all_properties)
    assert single.call_count == int(not all_properties)


def test_suite_roundtrip_with_real_offline_probes(tmp_path, monkeypatch, esm2_tokenizer):
    from accelerate import Accelerator
    from transformers import EsmConfig, EsmForMaskedLM

    from .conftest import build_raw_dataset_dict

    model_path = tmp_path / "checkpoint"
    EsmForMaskedLM(
        EsmConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=len(esm2_tokenizer),
            pad_token_id=esm2_tokenizer.pad_token_id,
            max_position_embeddings=128,
        )
    ).save_pretrained(model_path)
    esm2_tokenizer.save_pretrained(model_path)
    dataset = build_raw_dataset_dict("sequence_regression")
    for split, rows in dataset.items():
        rows = rows.add_column("id", list(range(len(rows))))
        for index, prop in enumerate(PROTPARAM_COLUMNS):
            rows = rows.add_column(
                prop, [(index + 1) * value + index for value in rows["targets"]]
            )
        dataset[split] = rows
    monkeypatch.setattr(
        "modules.evaluate.src.steps.prepare.resolve_evaluation_data",
        lambda *args, **kwargs: dataset,
    )
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "datasets/uniprot_protparams.yaml",
            CONFIGS / "predictors/sklearn_linear.yaml",
        ],
        overrides=[
            f"workspace.model_repo_id={model_path}",
            "workspace.model_name=tiny",
            f"workspace.output_dir={tmp_path / 'output'}",
            "steps.prepare.max_length=64",
            "steps.prepare.tokenize_num_proc=1",
            "steps.prepare.random_truncate_by_split.train=false",
            "steps.tune.hyperparameter_search.enabled=false",
            "steps.tune.head.search_space={}",
            "steps.predict.controls=[]",
            "steps.score.bootstrap_enabled=false",
        ],
    )
    summary = protparams.run_protparams(config, Accelerator(cpu=True))
    assert summary["n_properties"] == len(PROTPARAM_COLUMNS)
    root = config.workspace.dataset_root / "tiny"
    for prop in PROTPARAM_COLUMNS:
        prediction = pl.read_parquet(
            root / f"{config.variant}-run01-{prop}/seed_{config.seed}/predictions.parquet"
        )
        assert prediction.sort("id")["label"].to_list() == pytest.approx(dataset["test"][prop])
    assert protparams.run_protparams(config, Accelerator(cpu=True)) == summary
