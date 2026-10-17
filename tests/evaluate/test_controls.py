"""Control heads retain fitted preprocessing while randomizing learned weights."""

from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.utils.controls import ControlBuilder
from modules.evaluate.src.utils.embed import EmbeddingDataset


class _ControlTrunk(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.config = width
        self.projection = nn.Linear(width, width)


@pytest.mark.parametrize("control", ["untrained", "random_head", "random_both"])
@pytest.mark.parametrize("probe_hidden_size", [None, 3])
@pytest.mark.parametrize("standardize_features", [False, True])
def test_randomized_heads_preserve_training_feature_statistics(
    monkeypatch, control, probe_hidden_size, standardize_features
):
    trained_head = EmbeddingProbeHead(
        4,
        2,
        "sequence_classification",
        probe_hidden_size=probe_hidden_size,
        standardize_features=standardize_features,
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
