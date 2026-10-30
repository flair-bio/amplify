"""Offline CLI integration: cached frozen-probe jobs run through the generic engine.

Only network-bound seams are replaced (tokenizer/model download and dataset
hub loading). Tokenization, embedding extraction, probe training, scoring and
publication run for real on CPU. ``run_pipeline`` must not exist, so
the test fails if the legacy step pipeline handles the job.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from accelerate import Accelerator as HFAccelerator
from accelerate.state import AcceleratorState

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.artifacts.scores import LAYERWISE_COMPARISON_FILENAME
from modules.evaluate.src.dataset import resolution as resolution_module
from modules.evaluate.src.dataset.workspace import (
    get_evaluation_workspace,
    get_external_workspace,
    get_layer_workspace,
)
from modules.evaluate.src.predictors import probe_preparation as prepare_module
from modules.evaluate.src.utils.config_loader import load_evaluation_config

from .conftest import ModelFamily, build_raw_dataset_dict, build_tiny_model

#: Classical (gradient-free) heads fit on cached pooled embeddings.
CLASSICAL_HEADS = ("sklearn_linear",)


#: Supervised families on the generic engine: target/label format and class count
#: as their dataset carries them.
FAMILIES = {
    "sequence_classification": ("categorical", "scalar", 2),
    "sequence_regression": ("continuous", "scalar", 1),
    "token_classification": ("categorical", "per_token", 2),
    "sequence_multilabel_classification": ("categorical", "multilabel", 2),
}


def _write_config(
    tmp_path,
    task_type: str,
    *,
    mode: str = "tune_probe",
    layerwise: bool = False,
    external: bool = False,
    bootstrap: bool = False,
    head_type: str | None = None,
) -> str:
    target_type, label_format, num_labels = FAMILIES[task_type]
    steps: dict = {
        "mode": "tune_probe" if mode == "live" else mode,
        "tune": {
            "hyperparameter_search": {"enabled": False},
            "max_epochs": 2,
            "batch_size": 2,
        },
        "score": {
            "bootstrap_enabled": bootstrap,
            "n_samples": 20,
            "n_permutations": 20,
        },
    }
    if mode in {"finetune", "evaluate_as_is", "live"}:
        # Live trunk paths: no embedding cache (config validation requires it
        # for evaluate-as-is and finetune; live probes train on uncached inputs).
        steps["prepare"] = {"embedding_cache": {"enabled": False}}
    if mode == "finetune":
        steps["tune"]["head"] = {"head_type": "mlp"}
    if layerwise:
        steps["layerwise"] = {"enabled": True, "layers": [0, 2]}
        steps["tune"]["head"] = {"head_type": "torch_linear"}
    if head_type is not None:
        steps["tune"]["head"] = {"head_type": head_type}
    if head_type in CLASSICAL_HEADS:
        # Classical estimators are fit on fixed-size pooled embeddings only.
        steps["prepare"] = {"embedding_cache": {"enabled": True}}
    payload = {
        "seed": 3,
        "workspace": {
            "run_id": "offline",
            "model_name": "offline_model",
            "model_repo_id": "flair-bio/offline-model",
            "dataset_name": "offline_dataset",
            "dataset_repo_id": "biomap-research/offline_dataset",
            "task_type": task_type,
            "target_type": target_type,
            "label_format": label_format,
            "num_labels": None if task_type == "sequence_regression" else num_labels,
            "base_path": str(tmp_path / "eval"),
        },
        "steps": steps,
    }
    if external:
        payload["external_test_sets"] = [
            {"name": "heldout", "dataset_repo_id": "org/heldout"}
        ]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return str(path)


#: (task_type, layerwise, head_type, mode). Layerwise multilabel is outside the
#: layerwise validator's matrix, so only the flat multilabel run is listed for it.
CLI_CASES = (
    [(task_type, False, None, "tune_probe") for task_type in FAMILIES]
    + [
        (task_type, True, None, "tune_probe")
        for task_type in FAMILIES
        if task_type != "sequence_multilabel_classification"
    ]
    + [
        ("sequence_classification", False, "sklearn_linear", "tune_probe"),
        ("sequence_regression", False, "sklearn_linear", "tune_probe"),
    ]
    + [(task_type, False, None, "finetune") for task_type in FAMILIES]
    + [(task_type, False, None, "evaluate_as_is") for task_type in FAMILIES]
    + [
        ("sequence_classification", False, None, "live"),
        ("sequence_regression", False, None, "live"),
    ]
)


@pytest.mark.parametrize(
    ("task_type", "layerwise", "head_type", "mode"),
    CLI_CASES,
    ids=[
        f"{task}-{mode}-{'layerwise' if layer else 'flat'}-{head or 'default'}"
        for task, layer, head, mode in CLI_CASES
    ],
)
def test_frozen_probe_runs_through_generic_engine_not_legacy_pipeline(
    task_type,
    layerwise,
    head_type,
    mode,
    model_family: ModelFamily,
    monkeypatch,
    tmp_path,
) -> None:
    AcceleratorState._reset_state(reset_partial_state=True)
    config_path = _write_config(
        tmp_path,
        task_type,
        mode=mode,
        layerwise=layerwise,
        external=layerwise,
        bootstrap=layerwise,
        head_type=head_type,
    )

    assert not hasattr(run_evaluate, "run_pipeline"), "legacy pipeline must be retired"
    monkeypatch.setattr(
        run_evaluate,
        "Accelerator",
        lambda *args, **kwargs: HFAccelerator(cpu=True),
    )
    monkeypatch.setattr(
        prepare_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda model_id, **kwargs: model_family.tokenizer),
    )
    monkeypatch.setattr(
        prepare_module,
        "Accelerator",
        lambda *args, **kwargs: HFAccelerator(cpu=True),
    )
    monkeypatch.setattr(
        resolution_module,
        "load_evaluation_dataset",
        lambda repo_id, **kwargs: build_raw_dataset_dict(task_type),
    )
    # External sets resolve through the same Hub boundary; serve a local split.
    monkeypatch.setattr(
        resolution_module,
        "load_raw_evaluation_dataset",
        lambda repo_id, **kwargs: build_raw_dataset_dict(task_type),
    )
    _, _, num_labels = FAMILIES[task_type]
    num_labels = 1 if task_type == "sequence_regression" else num_labels

    class _FakeModelCls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return build_tiny_model(model_family, task_type, num_labels)

    monkeypatch.setattr(
        type(prepare_module.get_task(task_type)), "auto_model_class", _FakeModelCls
    )

    run_evaluate._run_model_evaluation([config_path], [], False)

    loaded = load_evaluation_config(path=[config_path])
    workspace = get_evaluation_workspace(
        loaded.workspace, variant=loaded.variant, seed=loaded.seed
    )
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    if not layerwise:
        assert any(workspace.run_root.glob("*.json")), (
            "scores/predictions must be written"
        )
        return
    external_workspace = get_external_workspace(
        workspace,
        SimpleNamespace(
            name="heldout",
            dataset_repo_id="org/heldout",
            dataset_revision=None,
            dataset_subset=None,
        ),
    )
    for view in (workspace, external_workspace):
        for layer in (0, 2):
            layer_workspace = get_layer_workspace(view, layer)
            assert any(layer_workspace.run_root.glob("*.json")), (
                f"layer {layer} predictions/scores must be written"
            )
        assert (view.comparison_path / LAYERWISE_COMPARISON_FILENAME).is_file(), (
            "paired cross-layer comparison must be written"
        )
