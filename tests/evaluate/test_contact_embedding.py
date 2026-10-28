"""Offline scientific-protocol and complete CASP embedding-pipeline tests."""

import json
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from transformers import EsmConfig, EsmModel

from modules.evaluate.src import run_evaluate
from modules.evaluate.src.dataset.workspace import get_evaluation_workspace
from modules.evaluate.src.model.contact_model import (
    BilinearHead,
    ContactPredictionModel,
    fit_contact_head,
)
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.tasks import available_tasks, get_task
from modules.evaluate.src.tasks.contact_prediction import (
    candidate_pairs,
    prepare_contact_dataset,
    tokenize_contact_chain,
)
from modules.evaluate.src.utils.config_loader import load_evaluation_config
from modules.evaluate.src.predictors.embeddings import contact_development_features
from modules.pretrain.src.model.modeling_amplify import register_amplify_auto_classes

from .conftest import build_tiny_model
from .test_contact_data import FEATURES, hf_data, row

CONFIGS = Path("modules/evaluate/configs")


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


def family_trunk(model_family):
    kwargs = dict(model_family.config_kwargs)
    kwargs["max_position_embeddings"] = 8
    if model_family.name == "esm2-35m":
        kwargs["position_embedding_type"] = "rotary"
    return build_tiny_model(
        model_family._replace(config_kwargs=kwargs),
        "sequence_classification",
        num_labels=2,
    ).base_model


def development_dataset():
    return DatasetDict(
        {
            split: Dataset.from_list(
                [
                    row(
                        f"{split}-{i}",
                        split,
                        sequence="ACDEFGHIK"[: 7 + i % 3],
                        positive_pairs=[],
                    )
                    for i in range(20)
                ],
                features=FEATURES,
            )
            for split in ("train", "validation", "test")
        }
    )


def config_for(tmp_path, model_path, seed=13):
    return load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "tasks/contact_prediction.yaml",
            CONFIGS / "datasets/contact_prediction_casp.yaml",
            CONFIGS / "predictors/contact_embedding.yaml",
        ],
        overrides=[
            f"workspace.output_dir={tmp_path / 'output'}",
            "workspace.run_id=offline",
            "workspace.model_name=tiny",
            f"workspace.model_repo_id={model_path}",
            f"workspace.dataset_repo_id={tmp_path / 'dataset'}",
            f"seed={seed}",
            "steps.tune.max_epochs=12",
            "steps.predict.controls=[random_trunk]",
        ],
    )


def test_head_directional_symmetry_and_pair_weighted_gradient():
    head = BilinearHead(3)
    first, second = torch.randn(7, 3), torch.randn(9, 3)
    directional = head.left(first) @ head.right(first).T
    torch.testing.assert_close(head(first), (directional + directional.T) / 2)
    scores = head(first)
    torch.testing.assert_close(scores, scores.T)
    scores.sum().backward()
    assert head.left.weight.grad is not None and head.right.weight.grad is not None

    accumulated, combined = deepcopy(head), deepcopy(head)
    accumulated.zero_grad()
    combined.zero_grad()
    values, labels = [], []
    for features in (first, second):
        i, j = np.triu_indices(len(features), 6)
        targets = torch.arange(len(i)).remainder(2).float()
        logits = accumulated(features)[i, j]
        F.binary_cross_entropy_with_logits(logits, targets, reduction="sum").div(
            7
        ).backward()
        values.append(combined(features)[i, j])
        labels.append(targets)
    F.binary_cross_entropy_with_logits(torch.cat(values), torch.cat(labels)).backward()
    for actual, expected in zip(accumulated.parameters(), combined.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_embedding_task_exposes_only_casp_metrics_and_rejects_flat_scoring():
    task = get_task("contact_prediction")
    assert "contact_prediction_embedding" not in available_tasks()
    assert task.default_metrics() == ["AUC", "P@L", "P@L/2", "P@L/5", "ROC-AUC"]
    task.validate_metric_names(task.default_metrics())
    with pytest.raises(KeyError):
        task.validate_metric_names(["contact_auc_long"])
    with pytest.raises(ValueError, match="Use score_units"):
        task.compute_metrics([], [], task.default_metrics())


def test_extraction_native_parity_gaps_growth_and_freezing(model_family):
    model = ContactPredictionModel(family_trunk(model_family))
    contacts = prepare_contact_dataset(hf_data([row(sequence="ACDEFGH")]))
    chain = contacts.splits["test"][0]
    tokenizer = model_family.tokenizer
    extracted = model.extract(chain, tokenizer)
    tokens = tokenizer(chain.alignment.clean_sequence, return_tensors="pt")
    with torch.no_grad():
        expected = model.trunk(**tokens).last_hidden_state[0, 1:-1]
    torch.testing.assert_close(extracted, expected)
    assert not any(parameter.requires_grad for parameter in model.trunk.parameters())
    assert extracted.shape == (7, 16) and not extracted.requires_grad

    contacts = prepare_contact_dataset(hf_data([row(sequence="ACXXDEFGHIK")]))
    gapped = contacts.splits["test"][0]
    first = model.extract(gapped, tokenizer)
    second = model.extract(chain, tokenizer)
    torch.testing.assert_close(second, expected)
    compressed = replace(
        gapped,
        alignment=replace(
            gapped.alignment,
            sequence=gapped.alignment.clean_sequence,
            retained_positions=tuple(range(9)),
            original_to_retained=tuple(range(9)),
        ),
    )
    assert not torch.allclose(first, model.extract(compressed, tokenizer))
    assert not model.trunk.training
    assert len(tokenize_contact_chain(gapped, tokenizer).input_ids) == 11
    longest = prepare_contact_dataset(
        hf_data(
            [
                row(sequence="A" * 3364, positive_pairs=[]),
            ]
        )
    ).splits["test"][0]
    assert model.extract(longest, tokenizer).shape == (3364, 16)


def test_fixed_fit_determinism_decay_patience_restoration_and_cap(monkeypatch, caplog):
    dataset = prepare_contact_dataset(development_dataset())
    model = ContactPredictionModel(small_trunk())
    generator = torch.Generator().manual_seed(7)
    features = {
        chain.record_id: torch.randn(
            len(chain.alignment.retained_positions), 8, generator=generator
        )
        for split in ("train", "validation")
        for chain in dataset.splits[split]
    }
    original = torch.optim.AdamW
    optimizers = []

    def optimizer(*args, **kwargs):
        result = original(*args, **kwargs)
        optimizers.append(result)
        return result

    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    settings = fit_contact_head(
        model, dataset, features, seed=88, max_epochs=12, device=torch.device("cpu")
    )
    assert settings["best_epoch"] == 1 and len(settings["history"]) == 11
    assert "Contact fit epoch 1/12 | train BCE=" in caplog.text
    assert "Contact fit early stopping at epoch 11/12" in caplog.text
    assert "restored_best_head=true" in caplog.text
    assert [group["weight_decay"] for group in optimizers[0].param_groups] == [10, 0]
    assert optimizers[0].param_groups[1]["params"] == [model.head.bias]
    first_state = deepcopy(model.head.state_dict())
    torch.rand(100)
    fit_contact_head(
        model, dataset, features, seed=88, max_epochs=12, device=torch.device("cpu")
    )
    for key, value in first_state.items():
        torch.testing.assert_close(value, model.head.state_dict()[key], rtol=0, atol=0)

    # All-negative validation maps are identically zero; state must be epoch 1.
    torch.manual_seed(88)
    reference = BilinearHead(8)
    opt = original(
        [
            {
                "params": [reference.left.weight, reference.right.weight],
                "weight_decay": 10,
            },
            {"params": [reference.bias], "weight_decay": 0},
        ],
        lr=0.001,
    )
    pairs = [candidate_pairs(c.units[0], c.alignment) for c in dataset.splits["train"]]
    total = sum(len(p.i) for p in pairs)
    for chain, pair in zip(dataset.splits["train"], pairs):
        F.binary_cross_entropy_with_logits(
            reference(features[chain.record_id])[pair.i, pair.j],
            torch.tensor(pair.labels).float(),
            reduction="sum",
        ).div(total).backward()
    opt.step()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(first_state[key], value)

    score = iter(range(1000))
    monkeypatch.setattr(
        "modules.evaluate.src.metrics.contact_metrics.score_contact_unit",
        lambda *args: {"P@L": next(score)},
    )
    with pytest.raises(RuntimeError, match="before validation patience"):
        fit_contact_head(
            model, dataset, features, seed=88, max_epochs=11, device=torch.device("cpu")
        )


def test_development_cache_is_seed_independent_and_never_contains_test(
    tmp_path, esm2_tokenizer
):
    dataset = prepare_contact_dataset(development_dataset())
    model = ContactPredictionModel(small_trunk())
    model.extract = Mock(wraps=model.extract)
    first, key = contact_development_features(model, esm2_tokenizer, dataset, tmp_path)
    assert model.extract.call_count == 40
    assert all(call.args[0].split != "test" for call in model.extract.call_args_list)
    model.extract.reset_mock()
    second, same_key = contact_development_features(
        model, esm2_tokenizer, dataset, tmp_path
    )
    assert same_key == key and model.extract.call_count == 0
    for record_id in first:
        torch.testing.assert_close(first[record_id], second[record_id])
    with torch.no_grad():
        next(model.trunk.parameters()).add_(1)
    _, changed_key = contact_development_features(
        model, esm2_tokenizer, dataset, tmp_path
    )
    assert changed_key != key and model.extract.call_count == 40


def test_offline_pipeline_checkpoint_and_seed_overwrite_isolation(
    tmp_path, model_family, monkeypatch
):
    register_amplify_auto_classes()
    tokenizer = model_family.tokenizer
    model_path = tmp_path / "trunk"
    family_trunk(model_family).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    hf = development_dataset()
    # Preserve a missing-label chain and an independently restricted EU output.
    hf["test"] = Dataset.from_list(
        [
            row("scored", positive_pairs=[[0, 6]]),
            row(
                "scored",
                kind="intra_eu",
                unit_id="D1",
                positions=[0, 6],
                valid_positions=[0, 6],
            ),
            row(
                "missing", labels_available=False, positive_pairs=[], valid_positions=[]
            ),
        ],
        features=FEATURES,
    )
    data = tmp_path / "dataset" / "data"
    data.mkdir(parents=True)
    for split, rows in hf.items():
        rows.to_parquet(data / f"{split}-00000-of-00001.parquet")
    config = config_for(tmp_path, model_path)
    workspace = get_evaluation_workspace(
        config.workspace, config.variant, seed=config.seed
    )
    results = []

    def run(*args, **kwargs):
        result = run_pipeline(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(run_evaluate, "load_evaluation_config", lambda **kwargs: config)
    monkeypatch.setattr(
        run_evaluate, "Accelerator", lambda **kwargs: Accelerator(cpu=True)
    )
    monkeypatch.setattr(run_evaluate, "run_pipeline", run)
    monkeypatch.setattr(sys, "argv", ["run_evaluate.py", "config.yaml"])
    run_evaluate.main()
    result = results[0]
    assert workspace.run_root.name == "seed_13"
    assert result.predictions[None].predictions.height == 3
    assert set(result.predictions[None].controls) == {"random_trunk"}
    assert result.scores[None].control_summaries["random_trunk"].height == 4
    assert (
        workspace.control_workspace("random_trunk").stats_path / "contact_training.json"
    ).is_file()
    assert (
        workspace.control_workspace("random_trunk").model_path / "contact_head.pt"
    ).is_file()
    assert (
        workspace.run_root / "controls" / "random_trunk" / "scores.parquet"
    ).is_file()
    assert workspace.task_type == "contact_prediction"
    assert workspace.variant == "bilinear512-frozen_trunk"
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["predict"]["metadata"]["head_type"] == "bilinear512"
    from modules.evaluate.src.steps.score import ScoreStep

    restored_predictions = ScoreStep.load_predict_output(workspace)
    assert restored_predictions.metadata.head_type == "bilinear512"
    assert set(restored_predictions.controls) == {"random_trunk"}
    assert (
        json.loads(workspace.run_manifest_path.read_text())["evaluation"]["status"]
        == "complete"
    )
    assert (workspace.stats_path / "contact_training.json").is_file()
    restored = ContactPredictionModel.from_pretrained(workspace.model_path)
    chain = prepare_contact_dataset(hf).splits["test"][0]
    predicted = restored.predict(chain, tokenizer).numpy()
    frame = result.predictions[None].predictions
    actual = frame.filter(frame["record_id"] == "scored").to_dicts()[0]
    np.testing.assert_allclose(
        actual["scores"],
        predicted[actual["original_i"], actual["original_j"]],
    )
    config2 = config_for(tmp_path, model_path, seed=14)
    sibling = get_evaluation_workspace(config2.workspace, config2.variant, seed=14)
    run_pipeline(config2, sibling, Accelerator(cpu=True))
    sibling_content = sibling.run_manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        run_pipeline(config, workspace, Accelerator(cpu=True))
    run_pipeline(config, workspace, Accelerator(cpu=True), overwrite=True)
    assert sibling.run_manifest_path.read_bytes() == sibling_content
    assert len(list(workspace.embeddings_path.glob("*.safetensors"))) == 3
    # Changing input weights cannot silently share an experiment across seeds.
    different = family_trunk(model_family)
    different.save_pretrained(model_path)
    with pytest.raises(ValueError, match="inputs differs across seeds"):
        run_pipeline(config, workspace, Accelerator(cpu=True), overwrite=True)


@pytest.mark.parametrize(
    "override",
    [
        "steps.tune.hyperparameter_search.enabled=true",
        "steps.tune.refit_on_train_validation=true",
        "steps.tune.standardize_features=true",
        "steps.tune.learning_rate=0.1",
        "steps.tune.head.head_type=mlp",
        "steps.prepare.packed=true",
        "steps.predict.controls=[random_head]",
        "steps.score.bootstrap_enabled=true",
    ],
)
def test_protocol_rejects_generic_settings(tmp_path, override):
    with pytest.raises(ValueError):
        load_evaluation_config(
            [
                CONFIGS / "config.yaml",
                CONFIGS / "datasets/contact_prediction_casp.yaml",
                CONFIGS / "predictors/contact_embedding.yaml",
            ],
            overrides=[override],
        )
