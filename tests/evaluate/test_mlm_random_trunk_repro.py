"""Reproducer: evaluate_as_is MLM jobs must produce the random_trunk condition.

Only the tokenizer/model/dataset Hub seams are replaced; the CLI, legacy
pipeline and control builder run for real on CPU.
"""

from __future__ import annotations

import json

import polars as pl
import pytest
import torch
from accelerate import Accelerator as HFAccelerator
from accelerate.state import AcceleratorState
from datasets import Dataset, DatasetDict

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.dataset import resolution as resolution_module
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.predictors import probe_preparation as prepare_module
from modules.evaluate.src.utils.amino_acids import CANONICAL_AMINO_ACIDS
from modules.evaluate.src.utils.config_loader import load_evaluation_config
from modules.evaluate.src.utils.controls import build_random_trunk
from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
from modules.pretrain.src.model.modeling_amplify import AMPLIFYForMaskedLM

from .conftest import LONG_SEQUENCE, SHORT_SEQUENCE


def _tiny_mlm(tokenizer):
    """Tiny AMPLIFY MLM standing in for a checkpoint.

    A seeded random init would coincide with the seeded random-trunk control,
    so shift every weight to make the primary a distinct, trained-like model.
    """
    model = _random_mlm(tokenizer)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.5)
    return model


def _random_mlm(tokenizer):
    config = AMPLIFYConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_position_embeddings=1300,
    )
    return AMPLIFYForMaskedLM(config)


#: Calibration windows span 512 residues around each scored position.
CALIBRATION_SEQUENCE = (LONG_SEQUENCE * 20)[:1200]
CALIBRATION_POSITIONS = [5, 600, 1190]


def _calibration_profile(position: int, consensus: str) -> list[float]:
    freqs = [0.0] * 20
    freqs[CANONICAL_AMINO_ACIDS.index(consensus)] = 1.0
    return [float(position), *freqs]


def _calibration_raw() -> DatasetDict:
    profiles = [
        [
            _calibration_profile(p, CALIBRATION_SEQUENCE[p])
            for p in CALIBRATION_POSITIONS
        ]
    ]
    return DatasetDict(
        {
            split: Dataset.from_dict(
                {
                    "id": ["P1"],
                    "sequence": [CALIBRATION_SEQUENCE],
                    "targets": profiles,
                }
            )
            for split in ("train", "validation", "test")
        }
    )


def _pseudo_perplexity_raw() -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_dict(
                {"sequence": [SHORT_SEQUENCE, LONG_SEQUENCE], "targets": [0, 0]}
            )
            for split in ("train", "validation", "test")
        }
    )


#: Per task: target/label contract, extra prepare config, and its dataset.
MLM_TASKS = {
    "pseudo_perplexity": {
        "target_type": "continuous",
        "label_format": "scalar",
        "prepare": {"embedding_cache": {"enabled": False}},
        "raw": _pseudo_perplexity_raw,
    },
    "calibration": {
        "target_type": "categorical",
        "label_format": "consensus_profile",
        "prepare": {
            "embedding_cache": {"enabled": False},
            "max_length": 100000,
        },
        "raw": _calibration_raw,
    },
}


@pytest.mark.parametrize("task_type", sorted(MLM_TASKS))
def test_evaluate_as_is_mlm_random_trunk_control_is_produced(
    task_type, model_family, monkeypatch, tmp_path
) -> None:
    AcceleratorState._reset_state(reset_partial_state=True)
    spec = MLM_TASKS[task_type]
    payload = {
        "seed": 3,
        "workspace": {
            "run_id": "offline",
            "model_name": "offline_model",
            "model_repo_id": "flair-bio/offline-model",
            "dataset_name": "offline_dataset",
            "dataset_repo_id": "biomap-research/offline_dataset",
            "task_type": task_type,
            "target_type": spec["target_type"],
            "label_format": spec["label_format"],
            "num_labels": None,
            "base_path": str(tmp_path / "eval"),
        },
        "steps": {
            "mode": "evaluate_as_is",
            "tune": {"hyperparameter_search": {"enabled": False}},
            "prepare": spec["prepare"],
            "predict": {"controls": ["random_trunk"]},
            "score": {"n_samples": 20, "n_permutations": 20},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload))

    raw = spec["raw"]()

    assert not hasattr(run_evaluate, "run_pipeline"), "legacy pipeline must be retired"
    monkeypatch.setattr(
        run_evaluate,
        "Accelerator",
        lambda *args, **kwargs: HFAccelerator(cpu=True),
    )
    monkeypatch.setattr(
        prepare_module.AutoTokenizer,
        "from_pretrained",
        staticmethod(
            lambda model_id, trust_remote_code=True, **kwargs: model_family.tokenizer
        ),
    )
    monkeypatch.setattr(
        prepare_module,
        "Accelerator",
        lambda *args, **kwargs: HFAccelerator(cpu=True),
    )
    monkeypatch.setattr(
        resolution_module,
        "load_evaluation_dataset",
        lambda repo_id, **kwargs: raw,
    )

    class _FakeModelCls:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            model = _tiny_mlm(model_family.tokenizer)
            primaries.append(
                (model, {k: v.detach().clone() for k, v in model.state_dict().items()})
            )
            return model

    primaries: list = []
    monkeypatch.setattr(
        type(prepare_module.get_task(task_type)), "auto_model_class", _FakeModelCls
    )

    run_evaluate._run_model_evaluation([str(config_path)], [], False)

    # The primary must never be re-initialized in place by the control.
    primary_model, primary_before = primaries[0]
    for name, tensor in primary_model.state_dict().items():
        torch.testing.assert_close(tensor, primary_before[name], msg=name)

    loaded = load_evaluation_config(path=[str(config_path)])
    workspace = get_evaluation_workspace(
        loaded.workspace, variant=loaded.variant, seed=loaded.seed
    )
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["evaluation"]["status"] == "complete"
    control_path = (
        workspace.run_root / "controls" / "random_trunk" / "predictions.parquet"
    )
    assert control_path.is_file(), sorted(
        str(p.relative_to(workspace.run_root)) for p in workspace.run_root.rglob("*")
    )
    primary = pl.read_parquet(workspace.run_root / "predictions.parquet")
    control = pl.read_parquet(control_path)
    assert control["id"].to_list() == primary["id"].to_list()
    assert not control["prediction"].equals(primary["prediction"]), (
        "random_trunk must re-initialize the trunk, not reuse the primary"
    )


def test_random_trunk_is_seed_deterministic_and_independent_of_primary(model_family):
    """The control is a pure function of (architecture, seed); the primary is untouched."""
    primary = _tiny_mlm(model_family.tokenizer)
    before = {k: v.detach().clone() for k, v in primary.state_dict().items()}

    first = build_random_trunk(primary, 3).state_dict()
    again = build_random_trunk(primary, 3).state_dict()
    other = build_random_trunk(primary, 4).state_dict()

    assert first.keys() == again.keys()
    for name in first:
        torch.testing.assert_close(first[name], again[name], msg=name)
    assert any(not torch.equal(first[n], other[n]) for n in first), (
        "a different seed must give different random weights"
    )
    for name, tensor in primary.state_dict().items():
        torch.testing.assert_close(tensor, before[name], msg=name)
    assert any(not torch.equal(first[n], before[n]) for n in before), (
        "random trunk must differ from the primary, not alias it"
    )
