"""Control probes use the correct features, training protocol, and preprocessing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.artifacts.cache import weights_fingerprint
from modules.evaluate.src.config import (
    HeadConfig,
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    PredictConfig,
    TuneConfig,
)
from modules.evaluate.src.config import EmbeddingCacheConfig
from modules.evaluate.src.controls.inference import (
    _get_or_build_control_embeddings,
    _run_embedding_control_inference,
)
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingDataset,
    find_embedding_dataset,
)
from modules.evaluate.src.dataset.types import CachedFeatures
from modules.evaluate.src.model.controls import build_random_trunk
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.inference import run_local_head_inference
from modules.evaluate.src.trainers.selection import _search_hyperparameters

from .helpers import (
    fit_controls,
    fit_inputs,
    make_inputs,
    patch_function,
    predict_inputs,
)
from .test_prepare import _make_workspace, _run_prepare


class _ControlTrunk(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.config = width
        self.projection = nn.Linear(width, width)


@pytest.mark.parametrize("control", ["untrained", "random_head", "random_both"])
@pytest.mark.parametrize("probe_hidden_size", [None, 3])
@pytest.mark.parametrize("standardize_features", [False, True])
@pytest.mark.parametrize("probe_hidden_depth", [1, 2])
def test_randomized_heads_preserve_training_feature_statistics(
    monkeypatch, control, probe_hidden_size, standardize_features, probe_hidden_depth
):
    trained_head = EmbeddingProbeHead(
        4,
        2,
        "sequence_classification",
        probe_hidden_size=probe_hidden_size,
        standardize_features=standardize_features,
        probe_hidden_depth=probe_hidden_depth,
    )
    if standardize_features:
        trained_head.fit_standardization(
            torch.tensor([[0.0, 10.0, 20.0, 30.0], [4.0, 14.0, 24.0, 34.0]])
        )
    with torch.no_grad():
        for parameter in trained_head.parameters():
            parameter.fill_(1.5)
    features = torch.tensor([[-2.0, -4.0, -6.0, -8.0], [1.0, 3.0, 5.0, 7.0]])
    labels = torch.tensor([0, 1])
    ids = ["a", "b"]
    prepared = make_inputs(
        model=trained_head,
        tokenizer=None,
        trunk=_ControlTrunk(4),
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=DataLoader(
            EmbeddingDataset(features, ids, labels), batch_size=2
        ),
        embedding=CachedFeatures(backend=SimpleNamespace()),
    )
    builder = PredictConfig(collect_probabilities=True)
    patch_function(
        monkeypatch,
        _get_or_build_control_embeddings,
        lambda *args: (features, ids, labels),
    )
    inference = run_local_head_inference
    control_heads = []

    def capture_head(head, *args, **kwargs):
        control_heads.append(head)
        return inference(head, *args, **kwargs)

    monkeypatch.setattr(
        "modules.evaluate.src.controls.inference.run_local_head_inference", capture_head
    )
    actual_ids, _, probabilities, _ = _run_embedding_control_inference(
        builder,
        control,
        SimpleNamespace(task_type="sequence_classification"),
        prepared.predictor,
        prepared.test,
        prepared.controls,
        Accelerator(cpu=True),
        seed=42,
    )
    assert actual_ids == ids
    (control_head,) = control_heads
    assert control_head is not trained_head
    assert control_head.probe_hidden_depth == trained_head.probe_hidden_depth
    assert control_head.probe_hidden_size == trained_head.probe_hidden_size
    assert set(control_head.state_dict()) == set(trained_head.state_dict())
    assert control_head.standardize_features is standardize_features
    if standardize_features:
        torch.testing.assert_close(control_head.feature_mean, trained_head.feature_mean)
        torch.testing.assert_close(control_head.feature_std, trained_head.feature_std)
        assert (
            control_head.feature_mean.data_ptr() != trained_head.feature_mean.data_ptr()
        )
        assert (
            control_head.feature_std.data_ptr() != trained_head.feature_std.data_ptr()
        )
    else:
        assert control_head.feature_mean is None
        assert control_head.feature_std is None
    assert any(
        (
            not torch.equal(parameter, trained_parameter)
            for parameter, trained_parameter in zip(
                control_head.parameters(), trained_head.parameters()
            )
        )
    )
    assert all((torch.all(parameter == 1.5) for parameter in trained_head.parameters()))
    with torch.inference_mode():
        expected = control_head(inputs_embeds=features).logits.softmax(dim=-1)
    torch.testing.assert_close(torch.tensor(probabilities), expected)


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize(
    "head_type,task_type",
    [
        ("mlp", "sequence_classification"),
        ("torch_linear", "token_classification"),
        ("torch_linear", "sequence_multilabel_classification"),
        ("torch_linear", "sequence_regression"),
        ("sklearn_linear", "sequence_classification"),
    ],
)
def test_cached_random_trunk_selects_and_refits_its_own_probe(
    monkeypatch, tmp_path, model_family, refit, head_type, task_type
):
    from modules.evaluate.src.controls import fitting as controls_module
    from modules.evaluate.src.trainers import fitting as fitting_module

    torch.manual_seed(1234)
    prepared = _run_prepare(
        monkeypatch,
        tmp_path,
        task_type,
        model_family,
        embedding_cache=EmbeddingCacheConfig(enabled=True),
    )
    workspace = _make_workspace(tmp_path, task_type)
    accelerator = Accelerator(cpu=True)
    original_loaders = (
        prepared.training.train_dataloader,
        prepared.training.val_dataloader,
        prepared.test.test_dataloader,
    )
    extracted = []
    original_build = controls_module.build_embedding_dataloaders

    def capture_features(**kwargs):
        assert set(kwargs["splits"]) == {"train", "validation"}
        loaders = original_build(**kwargs)
        extracted.append((kwargs["model"], loaders))
        return loaders

    monkeypatch.setattr(
        controls_module, "build_embedding_dataloaders", capture_features
    )
    searches = []
    original_search = _search_hyperparameters

    def capture_search(config, train_trunk, _fit_result, ws, *args, **kwargs):
        searches.append((ws, config.hyperparameter_search.model_dump()))
        return original_search(config, train_trunk, _fit_result, ws, *args, **kwargs)

    patch_function(monkeypatch, _search_hyperparameters, capture_search)
    original_fit = fitting_module.fit_predictor
    fit_requests = []

    def capture_fit(**kwargs):
        fit_requests.append(kwargs["workspace"])
        return original_fit(**kwargs)

    patch_function(monkeypatch, original_fit, capture_fit)
    config = TuneConfig(
        head=HeadConfig(
            head_type=head_type,
            hidden_size=3,
            search_space={"C": [0.1, 1.0]} if head_type == "sklearn_linear" else {},
        ),
        batch_size=2,
        max_epochs=1,
        refit_on_train_validation=refit,
        standardize_features=True,
        hyperparameter_search=HyperparameterSearchConfig(
            enabled=True,
            n_trials=2,
            search_pattern="grid",
            search_space=HyperparameterSearchSpaceConfig(
                learning_rate=[0.001, 0.01],
                weight_decay=[0.0],
                max_epochs=[1],
                hidden_size=[3, 5] if head_type == "mlp" else None,
                hidden_depth=[1, 2] if head_type == "mlp" else None,
            ),
        ),
    )
    fit_inputs(config, workspace, prepared, seed=42, accelerator=accelerator)
    fit_controls(
        config,
        workspace,
        prepared,
        seed=42,
        controls=["random_trunk"],
        accelerator=accelerator,
    )
    assert fit_requests == [
        workspace.config,
        workspace.control_workspace("random_trunk").config,
    ]
    assert prepared.controls.random_trunk_model is not prepared.predictor.model
    assert original_loaders == (
        prepared.training.train_dataloader,
        prepared.training.val_dataloader,
        prepared.test.test_dataloader,
    )
    ((trunk, loaders),) = extracted
    assert weights_fingerprint(trunk) == weights_fingerprint(
        build_random_trunk(prepared.predictor.trunk, 42)
    )
    train = find_embedding_dataset(loaders["train"])
    validation = find_embedding_dataset(loaders["validation"])
    fitted_features = (
        torch.cat([train.embeddings, validation.embeddings])
        if refit
        else train.embeddings
    )
    assert not torch.equal(
        train.embeddings,
        find_embedding_dataset(prepared.training.train_dataloader).embeddings,
    )
    if head_type == "sklearn_linear":
        np.testing.assert_allclose(
            prepared.controls.random_trunk_model.named_steps["scaler"].mean_,
            fitted_features.double().mean(0).numpy(),
            atol=1e-06,
        )
        filename = "tune_config.json"
    else:
        assert len(searches) == 2
        assert searches[0][1] == searches[1][1]
        assert searches[1][0] == workspace.config
        torch.testing.assert_close(
            prepared.controls.random_trunk_model.feature_mean.cpu(),
            fitted_features.float().mean(0),
        )
        filename = "probe_head_config.json"
    baseline_path = workspace.control_workspace("random_trunk").model_path
    settings = json.loads((baseline_path / filename).read_text())
    assert settings["final_fit_split"] == ("train+validation" if refit else "train")
    if head_type != "sklearn_linear":
        infer = run_local_head_inference

        def check_head(head, *args, **kwargs):
            assert head is prepared.controls.random_trunk_model
            return infer(head, *args, **kwargs)

        monkeypatch.setattr(
            "modules.evaluate.src.controls.inference.run_local_head_inference",
            check_head,
        )
    result = predict_inputs(
        PredictConfig(controls=["random_trunk"], persist_to_disk=False),
        workspace,
        prepared,
        seed=42,
        accelerator=accelerator,
    )
    assert (
        result.controls["random_trunk"]["id"].to_list()
        == result.predictions["id"].to_list()
    )
    with pytest.raises(ValueError, match="seed must match its tuning seed"):
        _run_embedding_control_inference(
            PredictConfig(collect_probabilities=False),
            "random_trunk",
            workspace,
            prepared.predictor,
            prepared.test,
            prepared.controls,
            accelerator,
            seed=43,
        )


def test_cached_random_trunk_requires_tuning():
    prepared = make_inputs(
        model=EmbeddingProbeHead(4, 2, "sequence_classification"),
        tokenizer=None,
        trunk=_ControlTrunk(4),
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        embedding=CachedFeatures(backend=SimpleNamespace()),
    )
    with pytest.raises(ValueError, match="requested controls before prediction"):
        _run_embedding_control_inference(
            PredictConfig(collect_probabilities=False),
            "random_trunk",
            SimpleNamespace(task_type="sequence_classification"),
            prepared.predictor,
            prepared.test,
            prepared.controls,
            Accelerator(cpu=True),
            seed=42,
        )
