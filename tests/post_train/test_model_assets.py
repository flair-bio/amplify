import json

import pytest

from modules.post_train.src.model.alignment import native_assets


@pytest.fixture
def checkpoint_assets(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "amplify",
        "auto_map": {
            "AutoConfig": "amplify.AMPLIFYConfig",
            "AutoModelForMaskedLM": "amplify.AMPLIFYForMaskedLM",
        },
    }))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "auto_map": {"AutoTokenizer": ["tokenizer.ProteinTokenizer", None]},
    }))
    (tmp_path / "amplify.py").write_text("from .layers import Layer\n")
    (tmp_path / "layers.py").write_text("from .norm import Norm\n")
    (tmp_path / "norm.py").write_text("class Norm: pass\n")
    (tmp_path / "tokenizer.py").write_text("class ProteinTokenizer: pass\n")
    (tmp_path / "unrelated.py").write_text("raise RuntimeError('Do not import')\n")
    return tmp_path


def test_discovers_transitive_model_and_tokenizer_assets_without_executing(checkpoint_assets):
    assert set(native_assets(checkpoint_assets)) == {
        "amplify.py", "layers.py", "norm.py", "tokenizer.py",
        "tokenizer.json", "tokenizer_config.json",
    }


@pytest.mark.parametrize("reference", [
    "owner/model--amplify.AMPLIFYForMaskedLM",
    "../amplify.AMPLIFYForMaskedLM",
    "missing.AMPLIFYForMaskedLM",
])
def test_rejects_nonlocal_or_missing_sources(checkpoint_assets, reference):
    config = json.loads((checkpoint_assets / "config.json").read_text())
    config["auto_map"]["AutoModelForMaskedLM"] = reference
    (checkpoint_assets / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="local|Missing"):
        native_assets(checkpoint_assets)


def test_rejects_missing_transitive_source(checkpoint_assets):
    (checkpoint_assets / "norm.py").unlink()
    with pytest.raises(ValueError, match="Missing native source dependency"):
        native_assets(checkpoint_assets)
