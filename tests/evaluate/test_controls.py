"""Control probes use the correct features, training protocol, and preprocessing."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.artifacts.feature_cache import (
    EmbeddingCacheConfig,
    weights_fingerprint,
)
from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.model.heads import EmbeddingProbeHead, HeadConfig
from modules.evaluate.src.predictors.embeddings import (
    EmbeddingDataset,
    find_embedding_dataset,
)
from modules.evaluate.src.predictors.inference import predict_trained
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.training.config import (
    HyperparameterSearchConfig,
    HyperparameterSearchSpaceConfig,
    TuneConfig,
)
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.training.tuning import fit_probe_protocol
from modules.evaluate.src.utils.controls import ControlBuilder, build_random_trunk

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
    prepared = PreparedArtifacts(
        model=trained_head,
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=DataLoader(
            EmbeddingDataset(features, ids, labels), batch_size=2
        ),
        embedding_backend=SimpleNamespace(trunk=_ControlTrunk(4)),
    )
    builder = ControlBuilder(collect_probabilities=True)
    monkeypatch.setattr(
        builder,
        "_get_or_build_control_embeddings",
        lambda *args: (features, ids, labels),
    )
    inference = builder._run_local_head_inference
    control_heads = []

    def capture_head(head, *args, **kwargs):
        control_heads.append(head)
        return inference(head, *args, **kwargs)

    monkeypatch.setattr(builder, "_run_local_head_inference", capture_head)

    actual_ids, _, probabilities, _ = builder._run_embedding_control_inference(
        control,
        SimpleNamespace(task_type="sequence_classification"),
        prepared,
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
        not torch.equal(parameter, trained_parameter)
        for parameter, trained_parameter in zip(
            control_head.parameters(), trained_head.parameters()
        )
    )
    assert all(torch.all(parameter == 1.5) for parameter in trained_head.parameters())
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
    from modules.evaluate.src.training import refit as refit_module
    from modules.evaluate.src.training.search_loop import search_hyperparameters

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
        prepared.train_dataloader,
        prepared.val_dataloader,
        prepared.test_dataloader,
    )
    extracted = []
    original_build = refit_module.build_embedding_dataloaders

    def capture_features(**kwargs):
        assert set(kwargs["splits"]) == {"train", "validation"}
        loaders = original_build(**kwargs)
        extracted.append((kwargs["model"], loaders))
        return loaders

    monkeypatch.setattr(refit_module, "build_embedding_dataloaders", capture_features)
    searches = []
    original_search = search_hyperparameters

    def capture_search(trainer, ws, *args, **kwargs):
        searches.append((ws.variant, trainer.config.hyperparameter_search.model_dump()))
        return original_search(trainer, ws, *args, **kwargs)

    monkeypatch.setattr(
        "modules.evaluate.src.training.tuning.search_hyperparameters", capture_search
    )
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
    fit_probe_protocol(
        TorchProbeTrainer(config),
        workspace,
        prepared,
        seed=42,
        controls=["random_trunk"],
        accelerator=accelerator,
    )
    assert prepared.random_trunk_model is not prepared.model
    assert original_loaders == (
        prepared.train_dataloader,
        prepared.val_dataloader,
        prepared.test_dataloader,
    )
    ((trunk, loaders),) = extracted
    backend = prepared.embedding_backend
    assert weights_fingerprint(trunk) == weights_fingerprint(
        build_random_trunk(backend.trunk, 42)
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
        find_embedding_dataset(prepared.train_dataloader).embeddings,
    )
    if head_type == "sklearn_linear":
        np.testing.assert_allclose(
            prepared.random_trunk_model.named_steps["scaler"].mean_,
            fitted_features.double().mean(0).numpy(),
            atol=1e-6,
        )
        filename = "tune_config.json"
    else:
        assert len(searches) == 2
        assert searches[0][1] == searches[1][1]
        assert searches[1][0] == workspace.variant
        torch.testing.assert_close(
            prepared.random_trunk_model.feature_mean.cpu(),
            fitted_features.float().mean(0),
        )
        filename = "probe_head_config.json"
    baseline_path = workspace.control_workspace("random_trunk").model_path
    settings = json.loads((baseline_path / filename).read_text())
    assert settings["final_fit_split"] == ("train+validation" if refit else "train")
    if head_type != "sklearn_linear":
        infer = ControlBuilder._run_local_head_inference

        def check_head(head, *args, **kwargs):
            assert head is prepared.random_trunk_model
            return infer(head, *args, **kwargs)

        monkeypatch.setattr(
            ControlBuilder, "_run_local_head_inference", staticmethod(check_head)
        )
    result = predict_trained(
        workspace,
        prepared,
        PredictConfig(controls=["random_trunk"], persist_to_disk=False),
        seed=42,
        accelerator=accelerator,
    )
    assert (
        result.controls["random_trunk"]["id"].to_list()
        == result.predictions["id"].to_list()
    )
    with pytest.raises(ValueError, match="seed must match its tuning seed"):
        ControlBuilder(False)._run_embedding_control_inference(
            "random_trunk", workspace, prepared, accelerator, seed=43
        )


def test_cached_random_trunk_requires_tuning():
    prepared = PreparedArtifacts(
        model=EmbeddingProbeHead(4, 2, "sequence_classification"),
        tokenizer=None,
        train_dataloader=None,
        val_dataloader=None,
        test_dataloader=None,
        embedding_backend=SimpleNamespace(trunk=_ControlTrunk(4)),
    )
    with pytest.raises(
        ValueError, match="pass the predict controls to fit_probe_protocol"
    ):
        ControlBuilder(False)._run_embedding_control_inference(
            "random_trunk",
            SimpleNamespace(task_type="sequence_classification"),
            prepared,
            Accelerator(cpu=True),
            seed=42,
        )
