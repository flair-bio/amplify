"""Dispatch of --baselines for structural tasks (contact, pseudo-perplexity).

Every Hub access is monkeypatched: these tests prove the network is touched only
at execution, through the declared loaders, and that offline validation is pure.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modules.evaluate.src.runners import baselines as runner
from modules.evaluate.src.runners import structural_baselines as sb
from modules.evaluate.src.tasks.contact_prediction import ContactPredictionTask

PINNED = "0123456789abcdef0123456789abcdef01234567"
REFERENCE_REVISION = "fedcba9876543210fedcba9876543210fedcba98"


def _config(task_type: str, names: list[str], **extra) -> runner.BaselineRunConfig:
    if task_type == "contact_prediction":
        shape = {
            "target_type": "categorical",
            "label_format": "contact_pairs",
            "num_labels": None,
        }
    else:  # PPL has no supervised labels; a continuous scalar is the valid shape.
        shape = {
            "target_type": "continuous",
            "label_format": "scalar",
            "num_labels": None,
        }
    workspace = {
        "base_path": "/tmp/eval-out",
        "task_type": task_type,
        "dataset_name": "fixture",
        "dataset_repo_id": "flair-bio/fixture",
        "dataset_revision": PINNED,
        **shape,
    }
    raw = {"workspace": workspace, "baselines": {"names": names}, **extra}
    if task_type == "pseudo_perplexity":
        raw["baselines"]["amino_acid_reference"] = {
            "repo_id": "flair-bio/reference",
            "revision": REFERENCE_REVISION,
        }
    return runner.BaselineRunConfig.model_validate(raw)


def test_structural_config_is_validated_offline(monkeypatch):
    """Constructing the config must not reach any loader."""

    def boom(*args, **kwargs):
        raise AssertionError("offline validation touched the network")

    monkeypatch.setattr(ContactPredictionTask, "prepare_dataset", boom)
    monkeypatch.setattr(sb, "load_amino_acid_reference", boom)
    _config("contact_prediction", ["separation_prior"])
    _config("pseudo_perplexity", ["amino_acid_frequency"])


def test_external_test_sets_are_rejected_for_structural_tasks():
    external = {"name": "x", "dataset_repo_id": "flair-bio/x", "split": "test"}
    with pytest.raises(ValidationError, match="external_test_sets are not supported"):
        _config(
            "contact_prediction",
            ["separation_prior"],
            external_test_sets=[external],
        )


def test_structural_task_rejects_sequence_baseline_names():
    with pytest.raises(ValueError, match="requires task_type"):
        _config("contact_prediction", ["train_mean"])


def test_contact_dispatch_runs_only_at_execution(tmp_path, monkeypatch):
    calls: dict[str, object] = {}
    dataset = object()

    def fake_prepare(self, source, *, revision, edition, cache_dir=None):
        calls["prepare"] = (source, revision, edition)
        return dataset

    def fake_run(*, dataset, dataset_identity, root, seed):
        calls["run"] = (dataset is not None, dataset_identity["revision"], seed)
        return root / "separation_prior"

    monkeypatch.setattr(ContactPredictionTask, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(sb, "run_contact_separation_baseline", fake_run)
    monkeypatch.setattr(
        "modules.evaluate.src.artifacts.identity.dataset_identity",
        lambda *, source, subset, revision: {"revision": revision},
    )
    config = _config("contact_prediction", ["separation_prior"], seed=11)
    outputs = runner.run_baselines(config)
    assert calls["prepare"] == ("flair-bio/fixture", PINNED, None)
    assert calls["run"] == (True, PINNED, 11)
    assert outputs["test"]["separation_prior"].name == "separation_prior"


def test_pseudo_perplexity_dispatch_reads_pinned_reference_and_test(
    tmp_path, monkeypatch
):
    seen: dict[str, object] = {}

    def fake_reference(reference, *, loader=None):
        seen["reference"] = (reference.repo_id, reference.revision, reference.split)
        return ["ACDE"]

    def fake_external(dataset_id, split, *, revision=None, subset=None):
        seen["test"] = (dataset_id, split, revision)
        return {"test": [{"id": "p1", "sequence": "ACD"}]}

    def fake_run(*, reference, reference_sequences, test_rows, dataset_identity, root):
        seen["rows"] = (list(reference_sequences), list(test_rows))
        return root / "amino_acid_frequency"

    monkeypatch.setattr(sb, "load_amino_acid_reference", fake_reference)
    monkeypatch.setattr(runner, "resolve_external_data", fake_external)
    monkeypatch.setattr(sb, "run_amino_acid_frequency_baseline", fake_run)
    config = _config("pseudo_perplexity", ["amino_acid_frequency"])
    outputs = runner.run_baselines(config)
    assert seen["reference"] == ("flair-bio/reference", REFERENCE_REVISION, "train")
    assert seen["test"] == ("flair-bio/fixture", "test", PINNED)
    assert seen["rows"] == (["ACDE"], [("p1", "ACD")])
    assert outputs["test"]["amino_acid_frequency"].name == "amino_acid_frequency"


def test_pseudo_perplexity_rejects_contact_names():
    with pytest.raises(ValueError, match="requires task_type"):
        _config("pseudo_perplexity", ["separation_prior"])


def test_token_and_multilabel_baselines_remain_rejected():
    shapes = {
        "token_classification": ("per_token", 3),
        "sequence_multilabel_classification": ("multilabel", 2),
    }
    for task_type, (label_format, num_labels) in shapes.items():
        with pytest.raises(ValueError, match="not supported"):
            runner.BaselineRunConfig.model_validate(
                {
                    "workspace": {
                        "base_path": "/tmp/eval-out",
                        "task_type": task_type,
                        "dataset_name": "fixture",
                        "dataset_repo_id": "flair-bio/fixture",
                        "dataset_revision": PINNED,
                        "target_type": "categorical",
                        "label_format": label_format,
                        "num_labels": num_labels,
                    },
                    "baselines": {"names": ["majority_class"]},
                }
            )
