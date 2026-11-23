import json
from types import SimpleNamespace

import pytest
import torch
from post_train_fixtures import BASE_VOCAB, write_tokenizer
from tokenizers import Tokenizer

from modules.post_train.src.model.alignment import load_student
from modules.post_train.src.model.tokenization import load_vocabulary


def test_checkpoint_owns_vocabulary_and_special_token_ids(tmp_path):
    vocab = BASE_VOCAB[1:] + BASE_VOCAB[:1] + ["<extra>"]
    checkpoint = write_tokenizer(tmp_path / "student", vocab)
    config_path = checkpoint / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["pad_token"] = {"content": "<pad>", "special": True}
    config_path.write_text(json.dumps(config))
    vocabulary = load_vocabulary(checkpoint, residues=set("ACDM"))
    tokens = {token: index for index, token in enumerate(vocab)}
    assert vocabulary.encode("ACDM") == [
        tokens["<bos>"], *[tokens[aa] for aa in "ACDM"], tokens["<eos>"],
    ]
    assert vocabulary.pad_id != 0
    assert vocabulary.mask_id == tokens["<mask>"]
    assert vocabulary.size == 33


def test_rejects_unsupported_residues_without_unknown_token_substitution(tmp_path):
    checkpoint = write_tokenizer(tmp_path / "student", [token for token in BASE_VOCAB if token != "C"])
    with pytest.raises(ValueError, match="does not contain residue"):
        load_vocabulary(checkpoint, residues=set("AC"))


def test_rejects_tokenizer_truncation(tmp_path):
    checkpoint = write_tokenizer(tmp_path / "student")
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=5)
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    with pytest.raises(ValueError, match="truncate"):
        load_vocabulary(checkpoint)


def test_rejects_special_token_metadata_that_disagrees_with_processor(tmp_path):
    checkpoint = write_tokenizer(tmp_path / "student")
    path = checkpoint / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["bos_token"], config["eos_token"] = config["eos_token"], config["bos_token"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="one token per residue"):
        load_vocabulary(checkpoint)


@pytest.mark.parametrize("changes", [
    {"vocab_size": 31}, {"pad_token_id": 1}, {"mask_token_id": 1},
    {"max_position_embeddings": 5},
])
def test_native_model_must_match_tokenizer_and_actual_input_capacity(tmp_path, monkeypatch, changes):
    from transformers import AutoModelForMaskedLM
    from modules.post_train.src.model import alignment

    checkpoint = write_tokenizer(tmp_path / "student")
    vocabulary = load_vocabulary(checkpoint)
    config = {
        "vocab_size": vocabulary.size, "pad_token_id": vocabulary.pad_id,
        "bos_token_id": vocabulary.bos_id, "eos_token_id": vocabulary.eos_id,
        "max_position_embeddings": 10, **changes,
    }
    native = SimpleNamespace(
        config=SimpleNamespace(**config), amplify=SimpleNamespace(norm=object()),
        lm_head=object(),
    )
    monkeypatch.setattr(alignment, "native_assets", lambda _: ())
    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", lambda *args, **kwargs: (native, {}))
    with pytest.raises(ValueError, match="interface or context capacity"):
        load_student(
            checkpoint, torch.device("cpu"), trust_remote_code=True,
            required_positions=10, vocabulary=vocabulary,
        )
