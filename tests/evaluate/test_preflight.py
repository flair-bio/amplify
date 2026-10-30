"""Execution preflight pins unpinned remote revisions; dry-run never resolves them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.workspace import EvaluationWorkspaceConfig
from modules.evaluate.src.evaluation import preflight

SHA_MODEL = "a" * 40
SHA_DATA = "b" * 40
SHA_EXTERNAL = "e" * 40
SHA_HUB_MAIN = "c" * 40


def _workspace(**overrides) -> EvaluationWorkspaceConfig:
    values = {
        "dataset_name": "toy_dataset",
        "run_id": "test",
        "dataset_repo_id": "flair-bio/toy-dataset",
        "model_name": "toy_model",
        "model_repo_id": "flair-bio/amplify-120m",
        "task_type": "sequence_classification",
        "target_type": "categorical",
        "label_format": "scalar",
        "num_labels": 2,
        "base_path": "/tmp/unused-eval-base",
    }
    values.update(overrides)
    return EvaluationWorkspaceConfig(**values)


class _FakeResolver:
    """Records every remote lookup so tests can prove what was (not) resolved."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, repo_id: str, kind: str) -> str:
        self.calls.append((repo_id, kind))
        return {"model": SHA_MODEL, "dataset": SHA_HUB_MAIN}[kind]


def test_unpinned_hub_model_and_dataset_are_resolved_to_commits() -> None:
    resolver = _FakeResolver()
    workspace = preflight.pin_model_revision(_workspace(), resolve=resolver)
    workspace, externals = preflight.pin_dataset_views(workspace, [], resolve=resolver)
    assert workspace.model_revision == SHA_MODEL
    assert workspace.dataset_revision == SHA_HUB_MAIN
    assert externals == []
    assert resolver.calls == [
        ("flair-bio/amplify-120m", "model"),
        ("flair-bio/toy-dataset", "dataset"),
    ]


def test_model_revision_pinned_on_workspace() -> None:
    resolver = _FakeResolver()
    pinned = preflight.pin_model_revision(_workspace(), resolve=resolver)
    assert pinned.model_revision == SHA_MODEL
    assert resolver.calls == [("flair-bio/amplify-120m", "model")]


def test_existing_pins_and_local_sources_are_not_resolved(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    resolver = _FakeResolver()
    workspace = _workspace(
        model_repo_id=str(checkpoint),
        dataset_repo_id=str(tmp_path / "local_dataset"),
        model_revision=None,
        dataset_revision=None,
    )
    pinned = preflight.pin_model_revision(workspace, resolve=resolver)
    pinned_data, _ = preflight.pin_dataset_views(pinned, [], resolve=resolver)
    assert pinned_data.model_revision is None
    assert pinned_data.dataset_revision is None
    assert resolver.calls == []

    already = _workspace(model_revision=SHA_MODEL, dataset_revision=SHA_DATA)
    preflight.pin_model_revision(already, resolve=resolver)
    preflight.pin_dataset_views(already, [], resolve=resolver)
    assert resolver.calls == []


def test_external_hub_views_resolve_only_when_unpinned(tmp_path) -> None:
    local_dir = tmp_path / "external_local"
    local_dir.mkdir()
    externals = [
        ExternalTestSetConfig(name="hub_unpinned", dataset_repo_id="org/hub_a"),
        ExternalTestSetConfig(
            name="hub_pinned",
            dataset_repo_id="org/hub_b",
            dataset_revision=SHA_EXTERNAL,
        ),
        ExternalTestSetConfig(name="local", dataset_repo_id=str(local_dir)),
    ]
    resolver = _FakeResolver()
    _, pinned = preflight.pin_dataset_views(
        _workspace(dataset_repo_id=None, dataset_name="toy", dataset_revision=SHA_DATA),
        externals,
        resolve=resolver,
    )
    assert [item.dataset_revision for item in pinned] == [
        SHA_HUB_MAIN,
        SHA_EXTERNAL,
        None,
    ]
    assert resolver.calls == [("org/hub_a", "dataset")]


def test_resolve_hub_commit_reads_default_sha_from_client(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeApi:
        def model_info(self, repo_id):
            calls.append(("model", repo_id))
            return SimpleNamespace(sha=SHA_MODEL)

        def dataset_info(self, repo_id):
            calls.append(("dataset", repo_id))
            return SimpleNamespace(sha=SHA_DATA)

    monkeypatch.setattr(preflight, "HfApi", FakeApi)
    assert preflight.resolve_hub_commit("org/model", "model") == SHA_MODEL
    assert preflight.resolve_hub_commit("org/data", "dataset") == SHA_DATA
    assert calls == [("model", "org/model"), ("dataset", "org/data")]


def test_resolve_hub_commit_failure_names_the_pinning_remedy(monkeypatch) -> None:
    class OfflineApi:
        def model_info(self, repo_id):
            raise OSError("network unreachable")

    monkeypatch.setattr(preflight, "HfApi", OfflineApi)
    with pytest.raises(RuntimeError, match="workspace.model_revision"):
        preflight.resolve_hub_commit("org/model", "model")


def test_resolve_hub_commit_rejects_missing_sha(monkeypatch) -> None:
    class NoShaApi:
        def dataset_info(self, repo_id):
            return SimpleNamespace(sha=None)

    monkeypatch.setattr(preflight, "HfApi", NoShaApi)
    with pytest.raises(RuntimeError, match="no 40-character commit SHA"):
        preflight.resolve_hub_commit("org/data", "dataset")


def test_dry_run_never_resolves_remote_revisions(monkeypatch, capsys) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not contact the Hub")

    monkeypatch.setattr(preflight, "resolve_hub_commit", forbidden)
    monkeypatch.setattr(preflight, "HfApi", forbidden)
    monkeypatch.setattr(run_evaluate, "pin_evaluation_config", forbidden)
    monkeypatch.setattr(run_evaluate, "pin_dataset_views", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_evaluate.py",
            "--suite",
            "calibration",
            "--model",
            "amplify_120m",
            "--dry-run",
        ],
    )
    run_evaluate.main()
    assert "1 run(s) validated" in capsys.readouterr().out


def test_immutable_pins_are_honored_offline(monkeypatch) -> None:
    """A full commit SHA on every remote source must never contact the Hub."""

    def forbidden(*args, **kwargs):
        raise AssertionError("pinned SHAs must be honored without network access")

    monkeypatch.setattr(preflight, "HfApi", forbidden)
    workspace = _workspace(model_revision=SHA_MODEL, dataset_revision=SHA_DATA)
    external = ExternalTestSetConfig(
        name="ext",
        dataset_repo_id="flair-bio/external",
        dataset_revision=SHA_EXTERNAL,
    )
    pinned_ws, pinned_ext = preflight.pin_dataset_views(
        preflight.pin_model_revision(workspace), [external]
    )
    assert pinned_ws.model_revision == SHA_MODEL
    assert pinned_ws.dataset_revision == SHA_DATA
    assert pinned_ext[0].dataset_revision == SHA_EXTERNAL


def test_hub_resolution_happens_only_at_execution_for_unpinned_refs(
    monkeypatch,
) -> None:
    """Unpinned refs are resolved by the injected execution resolver exactly once."""
    calls: list[tuple[str, str]] = []

    def record(repo_id: str, kind: str) -> str:
        calls.append((repo_id, kind))
        return SHA_MODEL if kind == "model" else SHA_DATA

    def forbidden(*args, **kwargs):
        raise AssertionError("resolution must use the injected resolver")

    monkeypatch.setattr(preflight, "HfApi", forbidden)
    workspace = preflight.pin_model_revision(_workspace(), resolve=record)
    workspace, _ = preflight.pin_dataset_views(workspace, [], resolve=record)
    assert calls == [
        ("flair-bio/amplify-120m", "model"),
        ("flair-bio/toy-dataset", "dataset"),
    ]
    assert workspace.model_revision == SHA_MODEL
    assert workspace.dataset_revision == SHA_DATA
