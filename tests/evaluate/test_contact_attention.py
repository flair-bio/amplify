"""Focused tests for the fixed CASP attention predictor."""

import torch
from transformers import EsmConfig, EsmModel

from modules.evaluate.src.model.contact_attention import (
    ATTENTION_C,
    ContactAttentionModel,
    fit_contact_attention_head,
)
from modules.evaluate.src.model.contact_model import load_contact_model
from modules.evaluate.src.tasks.contact_prediction import (
    ContactAlignment,
    ContactChain,
    ContactDataset,
    ContactUnit,
    candidate_pairs,
)


def small_amplify_trunk():
    from modules.pretrain.src.model.configuration_amplify import AMPLIFYConfig
    from modules.pretrain.src.model.modeling_amplify import AMPLIFYModel

    return AMPLIFYModel(
        AMPLIFYConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=30,
            pad_token_id=1,
            max_position_embeddings=32,
        )
    )


def small_trunk():
    return EsmModel(
        EsmConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            vocab_size=30,
            pad_token_id=1,
            position_embedding_type="rotary",
            max_position_embeddings=8,
            token_dropout=False,
        )
    )


def contact_chain(record_id, split, sequence="ACXXDEFGHIK"):
    retained = tuple(i for i, residue in enumerate(sequence) if residue != "X")
    clean = "".join(sequence[i] for i in retained)
    inverse = [-1] * len(sequence)
    for index, original in enumerate(retained):
        inverse[original] = index
    alignment = ContactAlignment(sequence, clean, retained, tuple(inverse))
    unit = ContactUnit(
        id=f"{record_id}:global",
        unit_id="global",
        kind="global",
        positions=None,
        labels_available=True,
        positive_pairs=((0, 6),) if int(record_id.rsplit("-", 1)[-1]) % 2 == 0 else (),
        valid_positions=retained,
    )
    return ContactChain(
        record_id,
        split,
        "trrosetta",
        None,
        None,
        None,
        alignment,
        (unit,),
    )


def test_attention_extraction_returns_all_layer_head_features(esm2_tokenizer):
    model = ContactAttentionModel(small_trunk())
    chain = contact_chain("test-0", "test")
    pairs = candidate_pairs(chain.units[0], chain.alignment)
    features = model.extract(chain, esm2_tokenizer, (pairs.i, pairs.j))
    assert features.shape == (2, len(pairs.i))
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_attention_extraction_handles_amplify_logits(esm2_tokenizer):
    model = ContactAttentionModel(small_amplify_trunk())
    chain = contact_chain("test-0", "test")
    pairs = candidate_pairs(chain.units[0], chain.alignment)

    features = model.extract(chain, esm2_tokenizer, (pairs.i, pairs.j))

    assert features.shape == (2, len(pairs.i))
    assert torch.isfinite(features).all()


def test_attention_head_fits_all_training_pairs_and_round_trips(
    tmp_path, esm2_tokenizer
):
    training = tuple(contact_chain(f"train-{i}", "train") for i in range(20))
    dataset = ContactDataset(
        {"train": training, "validation": (), "test": ()},
        "attention-test",
        "in-memory",
        None,
    )
    model = ContactAttentionModel(small_trunk())

    settings = fit_contact_attention_head(model, dataset, esm2_tokenizer, seed=3)

    assert settings["training_structures"] == 20
    assert settings["training_pairs"] == sum(
        len(candidate_pairs(chain.units[0], chain.alignment).i) for chain in training
    )
    assert settings["C"] == ATTENTION_C
    assert settings["solver"] == "liblinear"
    assert not settings["standardized"]
    prediction = model.predict(training[0], esm2_tokenizer)
    assert prediction.shape == (9, 9)
    torch.testing.assert_close(prediction, prediction.T)
    assert torch.isfinite(prediction).all()
    pairs = candidate_pairs(training[0].units[0], training[0].alignment)
    pair_features = model.extract(
        training[0], esm2_tokenizer, (pairs.i, pairs.j)
    ).T.numpy()
    assert model.classifier is not None
    expected = model.classifier.predict_proba(pair_features)[:, 1]
    torch.testing.assert_close(
        prediction[pairs.i, pairs.j], torch.tensor(expected, dtype=torch.float32)
    )

    model.save_pretrained(tmp_path)
    esm2_tokenizer.save_pretrained(tmp_path)
    restored = load_contact_model(tmp_path)
    assert isinstance(restored, ContactAttentionModel)
    restored_prediction = restored.predict(training[0], esm2_tokenizer)
    torch.testing.assert_close(prediction, restored_prediction)
    assert (tmp_path / "attention_head.joblib").is_file()
