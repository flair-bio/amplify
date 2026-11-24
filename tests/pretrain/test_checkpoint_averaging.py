"""Tests for `checkpoint_averaging` (uniform averaging of HF exports)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForMaskedLM

from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
from modules.pretrain.src.model.modeling_amplify import (
    AMPLIFYForMaskedLM,
    register_amplify_auto_classes,
)
from modules.pretrain.src.utils.checkpoint_averaging import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    average_checkpoints,
    main,
)


def _export(path, seed, **overrides):
    """An HF export as written by `Trainer.save_hf_checkpoint`."""
    torch.manual_seed(seed)
    config = AMPLIFYConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=27,
        **overrides,
    )
    model = AMPLIFYForMaskedLM(config)
    model.config.auto_map = register_amplify_auto_classes()
    model.save_pretrained(path)
    return path


@pytest.fixture
def checkpoints(tmp_path):
    paths = [_export(tmp_path / f"checkpoint_{s}", seed=s) for s in (1, 2, 3)]
    (paths[-1] / "tokenizer_config.json").write_text("{}")
    return paths


def test_writes_the_mean_and_a_loadable_checkpoint(checkpoints, tmp_path):
    output = tmp_path / "merged"
    main(["--checkpoints", *map(str, checkpoints), "--output-dir", str(output)])

    inputs = [load_file(p / WEIGHTS_NAME) for p in checkpoints]
    averaged = load_file(output / WEIGHTS_NAME)
    assert averaged.keys() == inputs[0].keys()
    for key, tensor in averaged.items():
        expected = torch.stack([w[key].double() for w in inputs]).mean(0)
        assert tensor.dtype == inputs[0][key].dtype
        torch.testing.assert_close(tensor, expected.to(tensor.dtype))

    # Non-weight files come from the last checkpoint.
    assert (output / "tokenizer_config.json").exists()
    assert (output / "modeling_amplify.py").exists()
    manifest = json.loads((output / MANIFEST_NAME).read_text())
    assert manifest["checkpoints"] == [str(p.resolve()) for p in checkpoints]

    model = AutoModelForMaskedLM.from_pretrained(
        output, trust_remote_code=True, local_files_only=True
    )
    reloaded = model.state_dict()
    assert reloaded.keys() == averaged.keys()
    assert all(torch.equal(reloaded[k], averaged[k]) for k in averaged)


def test_identical_inputs_are_preserved_exactly(tmp_path):
    paths = [_export(tmp_path / f"c{i}", seed=0) for i in range(3)]
    averaged = average_checkpoints(paths, tmp_path / "merged")
    expected = load_file(paths[0] / WEIGHTS_NAME)
    result = load_file(averaged / WEIGHTS_NAME)
    assert all(torch.equal(result[k], expected[k]) for k in expected)


def test_ignores_export_metadata_in_config(checkpoints, tmp_path):
    config_path = checkpoints[1] / "config.json"
    config = json.loads(config_path.read_text())
    config.update(_name_or_path="elsewhere", transformers_version="0.0.0")
    config_path.write_text(json.dumps(config))
    average_checkpoints(checkpoints, tmp_path / "merged")


def test_rejects_different_config(checkpoints, tmp_path):
    _export(checkpoints[1], seed=2, qk_norm=True)
    with pytest.raises(ValueError, match="config.json"):
        average_checkpoints(checkpoints, tmp_path / "merged")


def test_rejects_different_tensors(checkpoints, tmp_path):
    weights = load_file(checkpoints[1] / WEIGHTS_NAME)
    weights.pop(next(iter(weights)))
    save_file(weights, checkpoints[1] / WEIGHTS_NAME, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Tensor keys"):
        average_checkpoints(checkpoints, tmp_path / "merged")


def test_rejects_sharded_checkpoints(checkpoints, tmp_path):
    (checkpoints[0] / f"{WEIGHTS_NAME}.index.json").write_text("{}")
    with pytest.raises(ValueError, match="Sharded"):
        average_checkpoints(checkpoints, tmp_path / "merged")


@pytest.mark.parametrize("pick", [[0], [0, 0]])
def test_needs_two_distinct_checkpoints(checkpoints, tmp_path, pick):
    with pytest.raises(ValueError, match="two distinct"):
        average_checkpoints([checkpoints[i] for i in pick], tmp_path / "merged")


def test_refuses_existing_output(checkpoints, tmp_path):
    (tmp_path / "merged").mkdir()
    with pytest.raises(FileExistsError):
        average_checkpoints(checkpoints, tmp_path / "merged")
    assert not list((tmp_path / "merged").iterdir())
