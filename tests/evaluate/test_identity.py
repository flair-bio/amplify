"""Artifact identities: canonical digests, model fingerprints, dataset pinning/subsets."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from modules.evaluate.src.artifacts.identity import (
    dataset_identity,
    digest,
    file_tree_fingerprint,
    is_immutable_revision,
    model_identity,
    validate_dataset_pin,
    validate_model_pin,
)
from modules.evaluate.src.dataset.dataloader import (
    load_raw_evaluation_dataset,
)


def _write_split(root: Path, split: str, frame: pl.DataFrame) -> None:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(data / f"{split}-00000-of-00001.parquet")


def _write_subset(root: Path, columns: dict[str, list]) -> None:
    frame = pl.DataFrame(columns)
    for split in ("train", "validation", "test"):
        _write_split(root, split, frame)


# -- canonical digests -------------------------------------------------------


def test_digest_ignores_key_order_and_changes_with_values() -> None:
    assert digest({"a": 1, "b": [2, 3]}) == digest({"b": [2, 3], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    assert len(digest({"a": 1})) == 64


def test_digest_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError):
        digest({"a": float("nan")})


# -- model fingerprints ------------------------------------------------------


def test_file_tree_fingerprint_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for root in (first, second):
        (root / "sub").mkdir(parents=True)
        (root / "model.safetensors").write_bytes(b"weights-1")
        (root / "sub" / "config.json").write_text("{}")
    assert file_tree_fingerprint(first) == file_tree_fingerprint(second)

    (second / "model.safetensors").write_bytes(b"weights-2")
    assert file_tree_fingerprint(first) != file_tree_fingerprint(second)


def test_file_tree_fingerprint_binds_relative_paths(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "x.bin").write_bytes(b"same")
    (right / "y.bin").write_bytes(b"same")
    assert file_tree_fingerprint(left) != file_tree_fingerprint(right)


def test_file_tree_fingerprint_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_tree_fingerprint(tmp_path / "missing")


def test_model_identity_for_local_checkpoint_records_fingerprint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"w")
    identity = model_identity(
        label="local_model",
        source=str(checkpoint),
        requested_revision=None,
    )
    assert identity["label"] == "local_model"
    assert identity["local"] is True
    assert identity["fingerprint"] == file_tree_fingerprint(checkpoint)


def test_model_identity_for_hub_model_records_requested_revision() -> None:
    identity = model_identity(
        label="amplify_120m",
        source="chandar-lab/AMPLIFY_120M",
        requested_revision="a" * 40,
    )
    assert identity["local"] is False
    assert identity["requested_revision"] == "a" * 40
    assert identity["fingerprint"] is None


def test_model_identity_rejects_unsafe_label() -> None:
    with pytest.raises(ValueError):
        model_identity(label="../escape", source="x", requested_revision=None)


# -- dataset pinning ---------------------------------------------------------


def test_immutable_revision_requires_full_commit_sha() -> None:
    assert is_immutable_revision("a" * 40)
    assert not is_immutable_revision("main")
    assert not is_immutable_revision(None)


def test_dataset_identity_includes_subset_and_revision() -> None:
    base = dataset_identity(source="org/bench", subset=None, revision="b" * 40)
    other_subset = dataset_identity(
        source="org/bench", subset="stability", revision="b" * 40
    )
    other_revision = dataset_identity(
        source="org/bench", subset=None, revision="c" * 40
    )
    assert base["pinned"] is True
    assert base != other_subset
    assert base != other_revision
    assert digest(base) != digest(other_subset)


def test_dataset_identity_marks_unpinned_hub_source() -> None:
    identity = dataset_identity(source="org/bench", subset=None, revision=None)
    assert identity["pinned"] is False
    assert identity["revision"] is None


def test_dataset_identity_rejects_branch_as_revision() -> None:
    with pytest.raises(ValueError):
        dataset_identity(source="org/bench", subset=None, revision="main")


def test_dataset_identity_for_local_source_uses_content_fingerprint(
    tmp_path: Path,
) -> None:
    _write_subset(tmp_path, {"x": [1, 2], "y": ["a", "b"]})
    identity = dataset_identity(source=str(tmp_path), subset=None, revision=None)
    assert identity["local"] is True
    assert identity["content"] == file_tree_fingerprint(tmp_path)


def test_dataset_identity_local_subset_fingerprints_only_that_subset(
    tmp_path: Path,
) -> None:
    _write_subset(tmp_path / "a", {"x": [1]})
    _write_subset(tmp_path / "b", {"y": ["q"]})
    identity_a = dataset_identity(source=str(tmp_path), subset="a", revision=None)
    assert identity_a["content"] == file_tree_fingerprint(tmp_path / "a")
    (tmp_path / "b" / "data" / "test-00000-of-00001.parquet").write_bytes(b"changed")
    assert (
        dataset_identity(source=str(tmp_path), subset="a", revision=None)["content"]
        == identity_a["content"]
    )


# -- explicit subsets (two schemas in one local repo) ------------------------


def test_subsets_load_distinct_schemas_from_one_repo(tmp_path: Path) -> None:
    _write_subset(tmp_path / "stability", {"sequence": ["AA"], "stability": [0.5]})
    _write_subset(tmp_path / "solubility", {"sequence": ["CC"], "label": [1]})

    stability = load_raw_evaluation_dataset(tmp_path, subset="stability")
    solubility = load_raw_evaluation_dataset(tmp_path, subset="solubility")

    assert "stability" in stability["test"].column_names
    assert "label" not in stability["test"].column_names
    assert "label" in solubility["test"].column_names
    assert solubility["test"][0]["sequence"] == "CC"


def test_missing_subset_is_an_error(tmp_path: Path) -> None:
    _write_subset(tmp_path / "stability", {"sequence": ["AA"], "stability": [0.5]})
    with pytest.raises(FileNotFoundError):
        load_raw_evaluation_dataset(tmp_path, subset="absent")


def test_subset_rejects_unsafe_name(tmp_path: Path) -> None:
    _write_subset(tmp_path, {"sequence": ["AA"], "label": [1]})
    with pytest.raises(ValueError):
        load_raw_evaluation_dataset(tmp_path, subset="../escape")


# -- Hub revisions and subsets reach the loader ------------------------------


def test_hub_subset_and_revision_are_passed_to_the_loader(monkeypatch) -> None:
    from datasets import DatasetDict

    from modules.evaluate.src.dataset import dataloader

    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return DatasetDict()

    monkeypatch.setattr(dataloader, "load_dataset", fake_load_dataset)
    dataloader.load_raw_evaluation_dataset(
        "flair-bio/evaluation-benchmark", revision="a" * 40, subset="stability"
    )
    assert calls == [
        (
            "flair-bio/evaluation-benchmark",
            {"revision": "a" * 40, "name": "stability"},
        )
    ]


# -- run provenance refuses same-label different identities ------------------


def _identity_config(base_dir, **overrides):
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspaceConfig

    values = {
        "dataset_name": "toy_dataset",
        "run_id": "test",
        "dataset_repo_id": "flair-bio/toy-dataset",
        "dataset_revision": "b" * 40,
        "model_name": "toy_model",
        "model_repo_id": "flair-bio/amplify-120m",
        "model_revision": "a" * 40,
        "task_type": "sequence_classification",
        "target_type": "categorical",
        "label_format": "scalar",
        "num_labels": 2,
        "base_path": str(base_dir / "default"),
    }
    values.update(overrides)
    return EvaluationWorkspaceConfig(**values)


def test_same_labels_with_a_different_model_revision_are_refused(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace

    config = _identity_config(tmp_path)
    with EvaluationWorkspace(config, variant="mlp-frozen_trunk").reserve_run():
        pass
    changed = EvaluationWorkspace(
        config.model_copy(update={"model_revision": "c" * 40}),
        variant="torch_linear-frozen_trunk",
    )
    with (
        pytest.raises(ValueError, match="different model checkpoint"),
        changed.reserve_run(overwrite=True),
    ):
        pass


def test_same_labels_with_a_different_dataset_subset_are_refused(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace

    config = _identity_config(tmp_path, dataset_revision=None, dataset_subset="a")
    with EvaluationWorkspace(config, variant="mlp-frozen_trunk").reserve_run():
        pass
    changed = EvaluationWorkspace(
        config.model_copy(update={"dataset_subset": "b"}),
        variant="torch_linear-frozen_trunk",
    )
    with (
        pytest.raises(ValueError, match="different model checkpoint"),
        changed.reserve_run(overwrite=True),
    ):
        pass


def test_local_checkpoint_edit_is_refused_under_the_same_label(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"one")
    config = _identity_config(
        tmp_path,
        model_repo_id=str(checkpoint),
        model_revision=None,
        dataset_revision=None,
    )
    with EvaluationWorkspace(config, variant="mlp-frozen_trunk").reserve_run():
        pass
    (checkpoint / "model.safetensors").write_bytes(b"two")
    changed = EvaluationWorkspace(config, variant="torch_linear-frozen_trunk")
    with (
        pytest.raises(ValueError, match="different model checkpoint"),
        changed.reserve_run(overwrite=True),
    ):
        pass


def test_unchanged_identity_can_rerun_and_is_recorded(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace

    config = _identity_config(tmp_path)
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    with workspace.reserve_run():
        pass
    with EvaluationWorkspace(config, variant="torch_linear-frozen_trunk").reserve_run():
        pass
    recorded = json.loads(
        (workspace.workspace_root / "workspace_manifest.json").read_text()
    )
    assert recorded["identity"] == workspace.pinned_identity()
    assert recorded["identity"]["model"]["requested_revision"] == "a" * 40


def test_external_workspace_uses_its_own_revision_and_subset(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import (
        EvaluationWorkspace,
        get_external_workspace,
    )

    config = _identity_config(tmp_path, dataset_subset="main_view")
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    external = type(
        "External",
        (),
        {
            "name": "toy_external",
            "dataset_repo_id": "flair-bio/other",
            "dataset_revision": "d" * 40,
            "dataset_subset": "other_view",
        },
    )()
    identity = get_external_workspace(workspace, external).pinned_identity()
    assert identity["dataset"]["revision"] == "d" * 40
    assert identity["dataset"]["subset"] == "other_view"


# -- atomic JSON writes ------------------------------------------------------


def test_legacy_manifest_without_identity_is_refused_and_untouched(tmp_path) -> None:
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace

    config = _identity_config(tmp_path)
    workspace = EvaluationWorkspace(config, variant="mlp-frozen_trunk")
    workspace.workspace_root.mkdir(parents=True, exist_ok=True)
    manifest = workspace.workspace_root / "workspace_manifest.json"
    legacy = {
        "dataset_name": workspace.dataset_name,
        "dataset_repo_id": workspace.dataset_repo_id,
        "task_type": workspace.task_type,
        "model_name": workspace.model_name,
        "model_repo_id": workspace.model_repo_id,
    }
    manifest.write_text(json.dumps(legacy))
    before = manifest.read_bytes()

    with (
        pytest.raises(ValueError, match="predates checkpoint identity recording"),
        workspace.reserve_run(),
    ):
        pass
    assert manifest.read_bytes() == before
    assert not (workspace.run_root / "run_manifest.json").exists()


def test_failed_write_keeps_previous_json_and_leaves_no_temp_file(tmp_path) -> None:
    from modules.evaluate.src.utils.io import write_json

    target = tmp_path / "run_manifest.json"
    write_json(target, {"value": 1})
    with pytest.raises(TypeError):
        write_json(target, {"value": object()})
    assert json.loads(target.read_text()) == {"value": 1}
    assert [path.name for path in tmp_path.iterdir()] == ["run_manifest.json"]


SHA = "0123456789abcdef0123456789abcdef01234567"


def test_hub_contact_data_requires_pinned_revision() -> None:
    with pytest.raises(ValueError, match="pinned 40-character dataset_revision"):
        validate_dataset_pin(
            task_type="contact_prediction",
            repo_id="flair-bio/contact_prediction_casp",
            revision=None,
            subset=None,
        )
    validate_dataset_pin(
        task_type="contact_prediction",
        repo_id="flair-bio/contact_prediction_casp",
        revision=SHA,
        subset=None,
    )


def test_local_contact_data_needs_no_revision(tmp_path: Path) -> None:
    validate_dataset_pin(
        task_type="contact_prediction",
        repo_id=str(tmp_path),
        revision=None,
        subset=None,
    )
    with pytest.raises(ValueError, match="applies only to Hub datasets"):
        validate_dataset_pin(
            task_type="contact_prediction",
            repo_id=str(tmp_path),
            revision=SHA,
            subset=None,
        )


def test_contact_rejects_subsets() -> None:
    with pytest.raises(ValueError, match="does not support dataset subsets"):
        validate_dataset_pin(
            task_type="contact_prediction",
            repo_id="flair-bio/contact_prediction_casp",
            revision=SHA,
            subset="other",
        )


def test_dataset_revision_must_be_a_full_sha_offline() -> None:
    with pytest.raises(ValueError, match="40-character commit SHA"):
        validate_dataset_pin(
            task_type="sequence_regression",
            repo_id="flair-bio/meltome_mixed",
            revision="main",
            subset=None,
        )


def test_model_revision_rules_are_offline() -> None:
    validate_model_pin(repo_id="flair-bio/amplify-120m", revision=SHA)
    validate_model_pin(repo_id="flair-bio/amplify-120m", revision=None)
    with pytest.raises(ValueError, match="40-character commit SHA"):
        validate_model_pin(repo_id="flair-bio/amplify-120m", revision="main")
    with pytest.raises(ValueError, match="only to Hub models"):
        validate_model_pin(repo_id="/abs/checkpoint", revision=SHA)


def test_run_identity_binds_predictor_and_leaves_workspace_identity_shared(
    tmp_path,
) -> None:
    from modules.evaluate.src.config import EvaluationConfig
    from modules.evaluate.src.evaluation.reservation import run_identity

    base = EvaluationConfig(workspace=_identity_config(tmp_path))
    workspace_identity = {"model": {"label": "m"}, "dataset": {"label": "d"}}
    recorded = run_identity(base, workspace_identity)

    # Workspace identity is embedded unchanged: sibling predictors share it.
    assert {k: recorded[k] for k in workspace_identity} == workspace_identity
    assert recorded["predictor"]["mode"] == base.steps.mode
    assert recorded["predictor"]["head"]["head_type"] == "mlp"

    linear = base.model_copy(deep=True)
    linear.steps.tune.head.head_type = "torch_linear"
    other = run_identity(linear, workspace_identity)
    assert other["model"] == recorded["model"]
    assert other["predictor"]["id"] != recorded["predictor"]["id"]
    assert (
        run_identity(base, workspace_identity)["predictor"]["id"]
        == (recorded["predictor"]["id"])
    )


def test_predictor_identity_ignores_layerwise_settings_when_disabled(tmp_path) -> None:
    from modules.evaluate.src.config import EvaluationConfig
    from modules.evaluate.src.evaluation.reservation import run_identity

    base = EvaluationConfig(workspace=_identity_config(tmp_path))
    changed = base.model_copy(deep=True)
    changed.steps.layerwise.layers = [1, 2]
    assert (
        run_identity(base, {})["predictor"]["id"]
        == (run_identity(changed, {})["predictor"]["id"])
    )

    enabled = base.model_copy(deep=True)
    enabled.steps.layerwise.enabled = True
    enabled.steps.layerwise.layers = [1, 2]
    assert (
        run_identity(enabled, {})["predictor"]["id"]
        != (run_identity(base, {})["predictor"]["id"])
    )
