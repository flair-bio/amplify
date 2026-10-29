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
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.model.adapters import contact_positions
from modules.evaluate.src.model.contact_model import BilinearHead, ContactPredictionModel
from modules.evaluate.src.runners.pipeline import run_pipeline
from modules.evaluate.src.contracts.artifacts import ContactArtifacts, PreparedArtifacts
from modules.evaluate.src.tasks import available_tasks, get_task
from modules.evaluate.src.dataset.contact import candidate_pairs, prepare_contact_dataset, tokenize_contact_chain
from modules.evaluate.src.trainers.contact import fit_contact_head
from modules.evaluate.src.contracts.config_loader import load_evaluation_config
from modules.pretrain.src.model.modeling_amplify import (
    AMPLIFYConfig,
    AMPLIFYModel,
    register_amplify_auto_classes,
)

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


@pytest.mark.parametrize("predictor", ["attention_l1", "categorical_jacobian"])
@pytest.mark.parametrize("controls", [[], ["random_trunk"]])
def test_contact_predictor_pipeline_roundtrip(
    tmp_path, esm2_tokenizer, predictor, controls
):
    from transformers import EsmForMaskedLM

    from modules.evaluate.src.contracts.config import ScoreConfig
    from modules.evaluate.src.steps.score import ScoreStep

    model_path = tmp_path / "trunk"
    EsmForMaskedLM(small_trunk().config).save_pretrained(model_path)
    esm2_tokenizer.save_pretrained(model_path)
    hf = development_dataset()
    # Logistic fitting requires both positive and negative training pairs.
    hf["train"] = Dataset.from_list(
        [
            row(f"train-{i}", "train", sequence="ACDEFGHIK", positive_pairs=[[0, 6]])
            for i in range(20)
        ],
        features=FEATURES,
    )
    hf["test"] = Dataset.from_list([row(sequence="ACXXDEFGHIK")], features=FEATURES)
    preset = "contact_attention" if predictor == "attention_l1" else "contact_jacobian"
    config = load_evaluation_config(
        [
            CONFIGS / "config.yaml",
            CONFIGS / "datasets/contact_prediction_casp.yaml",
            CONFIGS / f"predictors/{preset}.yaml",
        ],
        overrides=[
            f"workspace.output_dir={tmp_path / 'output'}",
            "workspace.model_name=tiny",
            f"workspace.model_repo_id={model_path}",
            f"workspace.dataset_repo_id={tmp_path / 'dataset'}",
            "workspace.run_id=offline",
            "seed=13",
            f"steps.predict.controls={json.dumps(controls)}",
        ],
    )
    workspace = get_evaluation_workspace(
        config.workspace, config.variant, seed=config.seed
    )
    result = run_pipeline(config, workspace, Accelerator(cpu=True), dataset=hf)
    prediction = result.predictions[None]
    assert prediction.metadata.head_type == predictor
    assert set(prediction.controls) == set(controls)
    restored = ScoreStep.load_predict_output(workspace)
    scored = ScoreStep(ScoreConfig()).run(workspace, predict_result=restored)
    assert scored.summaries.equals(result.scores[None].summaries)
    assert set(scored.control_summaries) == set(controls)
    manifest = json.loads(workspace.run_manifest_path.read_text())
    assert manifest["predict"]["metadata"]["head_type"] == predictor
    assert manifest["evaluation"]["status"] == "complete"
    assert manifest["score"]["scores"] == scored.summaries.to_dicts()
    assert manifest["score"]["scores_path"] == str(scored.summaries_path)
    assert manifest["score"]["unit_scores_path"] == str(scored.unit_scores_path)
    assert set(manifest["score"]["controls"]) == set(controls)
    assert "bootstrap" not in manifest
    for name in controls:
        summary = manifest["score"]["controls"][name]
        assert summary["scores"] == scored.control_summaries[name].to_dicts()
        assert summary["scores_path"] == str(scored.control_summaries_paths[name])
        assert summary["unit_scores_path"] == str(scored.control_unit_scores_paths[name])
    if predictor == "categorical_jacobian":
        assert not (workspace.stats_path / "contact_training.json").exists()
    for name in controls:
        settings = json.loads(
            (
                workspace.control_workspace(name).stats_path / "contact_training.json"
            ).read_text()
        )
        assert settings["trunk_seed"] is not None
        if predictor == "categorical_jacobian":
            assert settings["fit"] == {"fitting": "zero_shot"}


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


def test_embedding_task_exposes_only_casp_metrics_and_structured_scoring():
    task = get_task("contact_prediction")
    assert "contact_prediction_embedding" not in available_tasks()
    assert task.default_metrics() == ["AUC", "P@L", "P@L/2", "P@L/5", "ROC-AUC"]
    task.validate_metric_names(task.default_metrics())
    with pytest.raises(KeyError):
        task.validate_metric_names(["contact_auc_long"])
    from modules.evaluate.src.tasks.base import StructuredTask
    from modules.evaluate.src.tasks.registry import get_flat_task

    assert isinstance(task, StructuredTask)
    assert not hasattr(task, "compute_metrics")
    with pytest.raises(TypeError, match="requires structured scoring"):
        get_flat_task("contact_prediction")


def test_prepared_contact_state_rejects_generic_loaders():
    dataset = prepare_contact_dataset(hf_data([row()]))
    with pytest.raises(ValueError, match="cannot be combined"):
        PreparedArtifacts(
            model=ContactPredictionModel(small_trunk()),
            tokenizer=None,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=torch.utils.data.DataLoader([1]),
            contact=ContactArtifacts(dataset=dataset),
        )


def test_extraction_native_parity_gaps_growth_and_freezing(model_family):
    model = ContactPredictionModel(family_trunk(model_family))
    contacts = prepare_contact_dataset(hf_data([row(sequence="ACDEFGH")]))
    chain = contacts.splits["test"][0]
    tokenizer = model_family.tokenizer
    extracted = model.extract(chain, tokenizer)
    tokens = tokenizer(chain.alignment.clean_sequence, return_tensors="pt")
    positions = torch.arange(tokens["input_ids"].shape[1]).unsqueeze(0)
    with torch.no_grad(), contact_positions(model.trunk, positions):
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


def test_positional_adapter_restores_tables_after_exception(model_family):
    trunk = family_trunk(model_family)
    saved = {
        (id(module), name): value
        for module in trunk.modules()
        for name in (
            "rope_cos",
            "rope_sin",
            "max_position_embeddings",
            "_cos_cached",
            "_sin_cached",
            "_seq_len_cached",
        )
        if (value := getattr(module, name, None)) is not None
    }
    with pytest.raises(RuntimeError, match="failed forward"):
        with contact_positions(trunk, torch.arange(20).unsqueeze(0)):
            raise RuntimeError("failed forward")
    for module in trunk.modules():
        for (identity, name), value in saved.items():
            if identity == id(module):
                assert getattr(module, name) is value


@pytest.mark.parametrize("initialized", [False, True])
def test_legacy_amplify_rotary_growth_and_restoration(initialized):
    trunk = AMPLIFYModel(
        AMPLIFYConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
    ).eval()
    rotary = trunk.rotary_emb
    rotary.reset_tables = None
    if not initialized:
        rotary.rope_cos = rotary.rope_sin = None
    old_cos, old_sin = rotary.rope_cos, rotary.rope_sin
    reference = deepcopy(trunk)
    reference.rotary_emb.max_position_embeddings = 22
    reference.rotary_emb._build_tables(torch.device("cpu"))
    inputs = {
        "input_ids": torch.tensor([[3, 5, 6, 7, 4]]),
        "position_ids": torch.tensor([[0, 1, 3, 12, 21]]),
    }
    with torch.no_grad():
        expected = reference(**inputs).last_hidden_state
        with contact_positions(trunk, inputs["position_ids"]):
            torch.testing.assert_close(trunk(**inputs).last_hidden_state, expected)
        assert rotary.max_position_embeddings == 8
        assert rotary.rope_cos is old_cos and rotary.rope_sin is old_sin
        with pytest.raises(RuntimeError, match="failed forward"):
            with contact_positions(trunk, inputs["position_ids"]):
                torch.testing.assert_close(trunk(**inputs).last_hidden_state, expected)
                raise RuntimeError("failed forward")
        assert rotary.max_position_embeddings == 8
        assert rotary.rope_cos is old_cos and rotary.rope_sin is old_sin


def test_legacy_amplify_rotary_growth_tolerates_device_rounding():
    trunk = AMPLIFYModel(
        AMPLIFYConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=8,
        )
    ).eval()
    rotary = trunk.rotary_emb
    rotary.reset_tables = None
    old_cos, old_sin = rotary.rope_cos.clone(), rotary.rope_sin.clone()
    build = rotary._build_tables

    def rounded_build(device, error=1e-6):
        build(device)
        rotary.rope_cos = rotary.rope_cos + error
        rotary.rope_sin = rotary.rope_sin - error

    positions = torch.tensor([[0, 1, 3, 12, 21]])
    rotary._build_tables = rounded_build
    with contact_positions(trunk, positions):
        assert torch.equal(rotary.rope_cos[:8], old_cos)
        assert torch.equal(rotary.rope_sin[:8], old_sin)
    assert torch.equal(rotary.rope_cos, old_cos)

    rotary._build_tables = lambda device: rounded_build(device, error=1e-2)
    with pytest.raises(ValueError, match="changed existing positions"):
        with contact_positions(trunk, positions):
            pass
    assert rotary.max_position_embeddings == 8
    assert torch.equal(rotary.rope_cos, old_cos)


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
    assert settings["effective_config"]["max_epochs"] == 12
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


def test_development_feature_identity_uses_only_development_metadata(esm2_tokenizer):
    from modules.evaluate.src.dataset.contact import tokenize_contact_chain
    from modules.evaluate.src.tasks.contact_prediction import (
        contact_development_feature_identity,
        contact_development_attention_fingerprint,
    )

    dataset = prepare_contact_dataset(development_dataset())
    development = [*dataset.splits["train"], *dataset.splits["validation"]]
    tokens = [tokenize_contact_chain(chain, esm2_tokenizer) for chain in development]
    metadata = dict(
        weights="weights",
        backbone_config={"hidden_size": 8},
        tf32=False,
        matmul_precision="highest",
    )
    identity = contact_development_feature_identity(dataset, tokens, **metadata)
    assert identity["extraction"] == "casp-final-embeddings-v1"
    assert {row[0] for row in identity["chains"]} == {"train", "validation"}
    assert len(identity["chains"]) == len(development)
    with pytest.raises(ValueError, match="token"):
        contact_development_feature_identity(dataset, tokens[:-1], **metadata)
    training_tokens = tokens[:len(dataset.splits["train"])]
    fingerprint = contact_development_attention_fingerprint(
        dataset, training_tokens, **metadata
    )
    changed = contact_development_attention_fingerprint(
        dataset, training_tokens, **{**metadata, "weights": "different"}
    )
    assert fingerprint != changed
    dataset.splits["test"] = ()
    assert contact_development_feature_identity(dataset, tokens, **metadata) == identity
    assert contact_development_attention_fingerprint(
        dataset, training_tokens, **metadata
    ) == fingerprint


def test_development_cache_is_seed_independent_and_never_contains_test(
    tmp_path, esm2_tokenizer
):
    dataset = prepare_contact_dataset(development_dataset())
    model = ContactPredictionModel(small_trunk())
    model.extract = Mock(wraps=model.extract)
    config = config_for(tmp_path, tmp_path / "trunk")
    workspace = get_evaluation_workspace(config.workspace, config.variant, seed=config.seed)
    workspace.setup_directories()
    prepared = PreparedArtifacts(
        model=model, tokenizer=esm2_tokenizer, train_dataloader=None,
        val_dataloader=None, test_dataloader=None, contact=ContactArtifacts(dataset),
    )
    step = PrepareStep(config.steps.prepare)
    step.prepare_features(workspace, prepared)
    first = prepared.features["development"].values
    key = prepared.features["development"].fingerprint
    assert model.extract.call_count == 40
    assert all(call.args[0].split != "test" for call in model.extract.call_args_list)
    model.extract.reset_mock()
    step.prepare_features(workspace, prepared)
    second = prepared.features["development"].values
    same_key = prepared.features["development"].fingerprint
    assert same_key == key and model.extract.call_count == 0
    for record_id in first:
        torch.testing.assert_close(first[record_id], second[record_id])
    with torch.no_grad():
        next(model.trunk.parameters()).add_(1)
    step.prepare_features(workspace, prepared)
    changed_key = prepared.features["development"].fingerprint
    assert changed_key != key and model.extract.call_count == 40


@pytest.mark.parametrize("materialize", [False, True])
def test_contact_tuning_requires_prepared_features_and_never_extracts(
    tmp_path, esm2_tokenizer, materialize
):
    from modules.evaluate.src.steps.tune import TuneStep

    config = config_for(tmp_path, tmp_path / "trunk")
    workspace = get_evaluation_workspace(config.workspace, config.variant, seed=config.seed)
    workspace.setup_directories()
    model = ContactPredictionModel(small_trunk())
    prepared = PreparedArtifacts(
        model=model, tokenizer=esm2_tokenizer,
        train_dataloader=None, val_dataloader=None, test_dataloader=None,
        contact=ContactArtifacts(prepare_contact_dataset(development_dataset())),
    )
    if materialize:
        PrepareStep(config.steps.prepare).prepare_features(workspace, prepared)
    model.extract = Mock(side_effect=AssertionError("Tune must not extract features"))
    step = TuneStep(config.steps.tune)
    if materialize:
        fitted = step.run(workspace, prepared, seed=config.seed, accelerator=Accelerator(cpu=True))
        assert fitted.model is model
        assert (workspace.stats_path / "contact_training.json").is_file()
    else:
        with pytest.raises(ValueError, match="prepared before fitting"):
            step.run(workspace, prepared, seed=config.seed, accelerator=Accelerator(cpu=True))
    model.extract.assert_not_called()


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
        "steps.tune.head.hyperparameters.learning_rate=0.1",
        "steps.tune.head.search_space.learning_rate=[0.1]",
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
