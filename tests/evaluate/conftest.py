"""Shared fixtures for modules.evaluate tests.

Covers both model families evaluated in this repo (see eval/):

- ``flair-bio/amplify-120m``: the custom ``ProteinTokenizer`` (character-level
  fast tokenizer, genuinely supports ``random_truncate``) + real
  ``AMPLIFYForSequenceClassification``/``AMPLIFYForTokenClassification``
  model classes (modules.pretrain.src.model.modeling_amplify), instantiated
  with tiny dimensions.
- ``facebook/esm2_t12_35M_UR50D``: the real HF ``EsmTokenizer`` (does *not*
  implement ``random_truncate``, exercising the manual-crop fallback path) +
  real ``EsmForSequenceClassification``/``EsmForTokenClassification`` model
  classes, instantiated with tiny dimensions.

Both are built entirely offline (no Hugging Face Hub access) from in-memory
vocabularies/configs, so tests exercise genuine tokenizer/model behavior
without network calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from datasets import Dataset, DatasetDict
from torch import nn
from transformers import EsmConfig, EsmTokenizer
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
    EsmForTokenClassification,
)

from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
from modules.pretrain.src.model.modeling_amplify import (
    AMPLIFYForSequenceClassification,
    AMPLIFYForTokenClassification,
)
from modules.pretrain.src.model.tokenizer import (
    ProteinTokenizer,
    TokenizerConfig,
    get_tokenizer,
)

import pytest

# Minimal but real amino-acid vocabulary (the 20 standard residues, enough to
# tokenize the toy sequences used below).
VOCAB = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<bos>",
    "<eos>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
]

# Real ESM2 vocabulary layout (facebook/esm2_t*): special tokens first, then
# the 20 standard residues plus common ambiguity codes.
ESM_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    "<mask>",
]

# A short and a long toy "protein" made only of residues present in VOCAB/ESM_VOCAB.
SHORT_SEQUENCE = "MKTAYIAKQR"
LONG_SEQUENCE = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTL"
)


@pytest.fixture()
def protein_tokenizer() -> ProteinTokenizer:
    """A small, real AMPLIFY fast tokenizer for protein sequences (no HF Hub access)."""
    cfg = TokenizerConfig(
        vocab=VOCAB,
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        bos_token="<bos>",
        eos_token="<eos>",
        max_length=512,
    )
    return get_tokenizer(cfg)


@pytest.fixture()
def esm2_tokenizer(tmp_path: Path) -> EsmTokenizer:
    """A real ``EsmTokenizer`` (facebook/esm2_* family), built offline.

    ``EsmTokenizer`` requires a vocab file on disk; unlike ``ProteinTokenizer``
    it does not implement ``random_truncate``, so this fixture exercises the
    ``tokenize_dataset`` manual-crop fallback path used for ESM2-family models.
    """
    vocab_file = tmp_path / "esm2_vocab.txt"
    vocab_file.write_text("\n".join(ESM_VOCAB))
    return EsmTokenizer(vocab_file=str(vocab_file))


class ModelFamily(NamedTuple):
    """A (tokenizer, tiny real model classes) bundle for one evaluated model family."""

    name: str
    tokenizer: ProteinTokenizer | EsmTokenizer
    sequence_model_cls: type[nn.Module]
    token_model_cls: type[nn.Module]
    config_kwargs: dict


@pytest.fixture(params=["amplify-120m", "esm2-35m"])
def model_family(request, protein_tokenizer, esm2_tokenizer) -> ModelFamily:
    """Parametrized fixture yielding both model families evaluated in this repo."""
    if request.param == "amplify-120m":
        return ModelFamily(
            name="amplify-120m",
            tokenizer=protein_tokenizer,
            sequence_model_cls=AMPLIFYForSequenceClassification,
            token_model_cls=AMPLIFYForTokenClassification,
            config_kwargs=dict(
                config_cls=AMPLIFYConfig,
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=32,
                vocab_size=len(VOCAB),
                pad_token_id=protein_tokenizer.pad_token_id,
                bos_token_id=protein_tokenizer.bos_token_id,
                eos_token_id=protein_tokenizer.eos_token_id,
                max_position_embeddings=256,
            ),
        )
    return ModelFamily(
        name="esm2-35m",
        tokenizer=esm2_tokenizer,
        sequence_model_cls=EsmForSequenceClassification,
        token_model_cls=EsmForTokenClassification,
        config_kwargs=dict(
            config_cls=EsmConfig,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            vocab_size=len(ESM_VOCAB),
            pad_token_id=esm2_tokenizer.pad_token_id,
            max_position_embeddings=256,
        ),
    )


def build_tiny_model(model_family: ModelFamily, task_type: str, num_labels: int):
    """Instantiate a tiny real model for *task_type* from *model_family*, offline."""
    kwargs = dict(model_family.config_kwargs)
    config_cls = kwargs.pop("config_cls")
    problem_type = "regression" if task_type == "sequence_regression" else None
    config = config_cls(num_labels=num_labels, problem_type=problem_type, **kwargs)
    model_cls = (
        model_family.token_model_cls
        if task_type == "token_classification"
        else model_family.sequence_model_cls
    )
    return model_cls(config)


def build_raw_dataset_dict(
    task_type: str,
    splits: tuple[str, ...] = ("train", "validation", "test"),
    num_labels: int = 2,
) -> DatasetDict:
    """Build a tiny, realistic (untokenized) DatasetDict for a given task type.

    - sequence_classification/regression: one scalar label per sequence.
    - token_classification: one label per residue (pre-tokenization, i.e. not
      yet accounting for BOS/EOS), matching what a raw HF eval dataset (e.g.
      biomap-research/*) would contain in its ``targets`` column.
    """
    sequences = [SHORT_SEQUENCE, LONG_SEQUENCE, SHORT_SEQUENCE[:5], LONG_SEQUENCE[:30]]

    if task_type == "token_classification":
        targets = [[i % num_labels for i in range(len(seq))] for seq in sequences]
    elif task_type == "sequence_regression":
        targets = [0.5, 1.25, -0.3, 2.0]
    else:
        targets = [0, 1, 0, 1]

    splits_dict = {}
    for split in splits:
        splits_dict[split] = Dataset.from_dict(
            {"sequence": sequences, "targets": targets}
        )
    return DatasetDict(splits_dict)
