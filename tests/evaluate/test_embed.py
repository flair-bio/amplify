"""Tests for modules.evaluate.src.utils.embed (frozen-trunk embedding cache).

Unit tests cover the storage/key primitives in isolation; the integration
tests reuse ``PrepareStep`` via ``tests.evaluate.test_prepare._run_prepare``
(offline-mocked tokenizer/dataset/model, same as the rest of test_prepare.py)
to verify the end-to-end cache-enabled Prepare flow with real tiny models.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from modules.evaluate.src.config import EvaluationConfig, LayerwiseConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel, resolve_layers
from modules.evaluate.src.utils.layerwise import run_layerwise_conditions
from modules.evaluate.src.utils.embed import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    EmbeddingDataset,
    embedding_collate_fn,
    extract_embeddings,
    embedding_cache_key,
    pool_hidden_state_layers,
    pool_hidden_states,
    residue_attention_mask,
)

from .test_prepare import _run_prepare

import pytest


@pytest.mark.parametrize("config_type", [LayerwiseConfig, EmbeddingCacheConfig])
def test_probe_normalization_defaults_and_override(
    config_type: type[LayerwiseConfig] | type[EmbeddingCacheConfig],
) -> None:
    assert config_type().normalize_before_pooling is True
    assert config_type(normalize_before_pooling=False).normalize_before_pooling is False


def test_normalized_pooling_is_invariant_to_representation_scale() -> None:
    states = torch.tensor([[[3.0, 4.0], [4.0, 3.0]]])
    mask = torch.ones((1, 2), dtype=torch.long)
    pooled = pool_hidden_states(states, mask, "mean", normalize_before_pooling=True)
    scaled = pool_hidden_states(
        states * 1000, mask, "mean", normalize_before_pooling=True
    )
    torch.testing.assert_close(pooled, scaled)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_cached_and_online_pooling_exclude_boundary_tokens(
    packed: bool, normalize: bool
) -> None:
    tokenizer = SimpleNamespace(
        cls_token_id=99, eos_token_id=98, pad_token_id=0,
        batch_decode=lambda rows, **kwargs: ["" for _ in rows],
    )

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(hidden_size=1, num_hidden_layers=1)

        def forward(self, input_ids, **kwargs):
            state = input_ids.unsqueeze(-1).float() * self.weight
            return SimpleNamespace(
                last_hidden_state=state + 1, hidden_states=(state, state + 1)
            )

    trunk = Trunk()
    batch = {"labels": torch.tensor([0.0, 1.0]), "id": ["first", "second"]}
    if packed:
        batch.update(
            input_ids=torch.tensor([[99, 2, 4, 98, 99, 8, 98, 0]]),
            cu_seqlens=torch.tensor([0, 4, 7, 8]), num_sequences=2,
        )
    else:
        batch.update(
            input_ids=torch.tensor([[99, 2, 4, 98], [99, 8, 98, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), num_processes=1,
        gather_for_metrics=lambda values, **kwargs: values,
    )
    cached, _, _, _ = extract_embeddings(
        trunk, [batch], accelerator, "mean", "float32", layers=[0, 1], tokenizer=tokenizer,
        normalize_before_pooling=normalize,
    )
    expected = torch.ones((2, 2)) if normalize else torch.tensor([[3.0, 4.0], [8.0, 9.0]])
    assert torch.equal(cached, expected)
    probe = LayerwiseProbeModel(
        SimpleNamespace(base_model=trunk, config=trunk.config), 1, "sequence_regression",
        LayerwiseConfig(enabled=True, layers=[0, 1], normalize_before_pooling=normalize),
        tokenizer=tokenizer,
    )
    for head in probe.heads.values():
        nn.init.ones_(head.classifier.weight)
        nn.init.zeros_(head.classifier.bias)
    online = probe(**batch)
    for index, output in online.items():
        assert torch.equal(output.logits[:, 0], cached[:, index])


class LocalLayerwiseAccelerator:
    num_processes = 1

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def gather_for_metrics(values, **kwargs):
        return values


@pytest.mark.parametrize(
    "task_type", ["sequence_classification", "token_classification"]
)
def test_online_layerwise_heads_share_frozen_trunk(model_family, task_type):
    from .conftest import build_tiny_model

    model = build_tiny_model(model_family, task_type, 2)
    probe = LayerwiseProbeModel(
        model,
        2,
        task_type,
        LayerwiseConfig(enabled=True, layers=[0, -1], autocast_dtype="float32"),
        tokenizer=model_family.tokenizer,
    )
    batch = {
        "input_ids": torch.tensor([[3, 5, 6, 4], [3, 5, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": (
            torch.tensor([0, 1])
            if task_type == "sequence_classification"
            else torch.tensor([[-100, 0, 1, -100], [-100, 1, -100, -100]])
        ),
    }
    output = probe(**batch)
    assert set(output) == {0, 2}
    assert output[0].logits.shape == (
        (2, 4, 2) if task_type == "token_classification" else (2, 2)
    )
    sum(item.loss for item in output.values()).backward()
    assert all(parameter.grad is None for parameter in probe.trunk.parameters())
    assert all(head.classifier.weight.grad is not None for head in probe.heads.values())
    assert resolve_layers([0, -1], 3) == [0, 2]
    with pytest.raises(ValueError, match="same hidden state"):
        resolve_layers([2, -1], 3)


def test_online_layerwise_token_inference_skips_ignored_positions(model_family):
    from .conftest import build_tiny_model

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, "token_classification", 2),
        2,
        "token_classification",
        LayerwiseConfig(enabled=True, layers=[0, -1], autocast_dtype="float32"),
        tokenizer=model_family.tokenizer,
    )
    batch = {
        "input_ids": torch.tensor([[3, 5, 6, 4], [3, 5, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, 0, 1, -100], [-100, 1, -100, -100]]),
        "id": ["first", "second"],
    }
    results = run_layerwise_conditions(
        probe, [batch], "token_classification", LocalLayerwiseAccelerator()
    )["trained"]
    assert set(results) == {0, 2}
    for ids, predictions, labels, _, _ in results.values():
        assert ids == ["first", "second"]
        assert labels == [[0, 1], [1]]
        assert list(map(len, predictions)) == [2, 1]


@pytest.mark.parametrize("pooling", ["mean", "cls"])
@pytest.mark.parametrize("normalize", [False, True])
def test_layerwise_packed_pooling_matches_padded(pooling, normalize):
    tokenizer = SimpleNamespace(cls_token_id=99, eos_token_id=98, pad_token_id=0)

    class Trunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=3)
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, output_hidden_states=False, **kwargs):
            state = input_ids.unsqueeze(-1).float().expand(-1, -1, 3) * self.weight
            state = state + torch.arange(3.0)
            return SimpleNamespace(hidden_states=(state, state * 2))

    trunk = Trunk()
    probe = LayerwiseProbeModel(
        SimpleNamespace(base_model=trunk, config=trunk.config), 1, "sequence_regression",
        LayerwiseConfig(
            enabled=True, layers="all", autocast_dtype="float32", pooling=pooling,
            normalize_before_pooling=normalize,
        ),
        tokenizer=tokenizer,
    )
    padded = {
        "input_ids": torch.tensor([[99, 2, 4, 98], [99, 8, 98, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }
    packed = {
        "input_ids": torch.tensor([[99, 2, 4, 98, 99, 8, 98, 0]]),
        "cu_seqlens": torch.tensor([0, 4, 7, 8], dtype=torch.int32),
        "num_sequences": 2,
    }
    expected, _, _ = probe._extract_features(**padded)
    actual, _, _ = probe._extract_features(**packed)
    for layer in probe.layers:
        torch.testing.assert_close(actual[layer], expected[layer])


def test_layerwise_cached_input_skips_trunk(model_family):
    from .conftest import build_tiny_model

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, "sequence_classification", 2),
        2,
        "sequence_classification",
        LayerwiseConfig(enabled=True, layers=[0, -1]),
        tokenizer=model_family.tokenizer,
    )
    features = torch.randn(3, probe.embedding_size * 2)
    probe.trunk.forward = MagicMock(side_effect=AssertionError("unexpected trunk pass"))
    outputs = probe(inputs_embeds=features, labels=torch.tensor([0, 1, 0]))
    for position, layer in enumerate(probe.layers):
        expected = probe.heads[str(layer)](
            inputs_embeds=features[
                :,
                position * probe.embedding_size : (position + 1) * probe.embedding_size,
            ]
        ).logits
        assert torch.allclose(outputs[layer].logits, expected)
    with pytest.raises(ValueError, match="wrong width"):
        probe(inputs_embeds=features[:, :-1])


def test_layerwise_standardization_removes_feature_and_target_scale(model_family):
    from .conftest import build_tiny_model

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, "sequence_regression", 1),
        1,
        "sequence_regression",
        LayerwiseConfig(enabled=True, layers=[0, -1]),
        tokenizer=model_family.tokenizer,
    )
    features = torch.randn(8, probe.embedding_size * 2) * 50 + 10
    labels = torch.linspace(100.0, 200.0, 8)

    def standardized_logits(scale: float) -> dict[int, torch.Tensor]:
        batch = {"inputs_embeds": features * scale, "labels": labels}
        probe.fit_standardization([batch], LocalLayerwiseAccelerator())
        return {layer: output.logits for layer, output in probe(**batch).items()}

    small, large = standardized_logits(1.0), standardized_logits(1000.0)
    for layer in probe.layers:
        torch.testing.assert_close(small[layer], large[layer], rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(
        probe.feature_mean[0], (features * 1000.0)[:, : probe.embedding_size].mean(0)
    )
    assert probe.target_mean.item() == pytest.approx(150.0)
    for head in probe.heads.values():
        nn.init.zeros_(head.classifier.weight)
        nn.init.zeros_(head.classifier.bias)
    outputs = probe(inputs_embeds=features * 1000.0, labels=labels)
    for output in outputs.values():
        torch.testing.assert_close(output.logits[:, 0], torch.full((8,), 150.0))
        assert output.loss.item() == pytest.approx(1.0, rel=1e-4)


def test_layerwise_token_standardization_ignores_unlabeled_positions(model_family):
    from .conftest import build_tiny_model

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, "token_classification", 2),
        2,
        "token_classification",
        LayerwiseConfig(enabled=True, layers=[0, -1]),
        tokenizer=model_family.tokenizer,
    )
    tokens = {
        "input_ids": torch.tensor([[3, 5, 6, 4], [3, 5, 4, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }
    labels = torch.tensor([[-100, 0, 1, -100], [-100, 1, -100, -100]])
    probe.fit_standardization(
        [dict(tokens, labels=labels)], LocalLayerwiseAccelerator()
    )
    states, _, _ = probe._extract_features(**tokens)
    for position, layer in enumerate(probe.layers):
        torch.testing.assert_close(
            probe.feature_mean[position], states[layer][labels != -100].mean(0)
        )


@pytest.mark.parametrize("task_type", ["sequence_classification", "token_classification"])
def test_distributed_layerwise_deduplicates_examples(model_family, task_type):
    from .conftest import build_tiny_model

    class Accelerator(LocalLayerwiseAccelerator):
        num_processes = 2

        @staticmethod
        def pad_across_processes(tensor, **kwargs):
            return tensor

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, task_type, 2), 2, task_type,
        LayerwiseConfig(enabled=True, layers=[0, -1]), tokenizer=model_family.tokenizer,
    )
    batch = {
        "input_ids": torch.tensor([[3, 5, 6, 4]] * 3),
        "attention_mask": torch.ones(3, 4),
        "labels": (
            torch.tensor([0, 1, 0]) if task_type == "sequence_classification" else
            torch.tensor([[-100, 0, 1, -100]] * 3)
        ),
        "id": ["first", "second", "first"],
    }
    results = run_layerwise_conditions(
        probe, [batch, dict(batch, id=["third", "first", "second"])], task_type,
        Accelerator(), collect_probabilities=task_type == "sequence_classification",
    )["trained"]
    for ids, predictions, labels, probabilities, _ in results.values():
        assert ids == ["first", "second", "third"]
        assert len(predictions) == len(labels) == 3
        if probabilities is not None:
            assert len(probabilities) == 3


def test_layerwise_multilabel_probabilities_are_independent(model_family):
    from .conftest import build_tiny_model

    probe = LayerwiseProbeModel(
        build_tiny_model(model_family, "sequence_multilabel_classification", 2),
        2,
        "sequence_multilabel_classification",
        LayerwiseConfig(enabled=True, layers=[0, -1]),
        tokenizer=model_family.tokenizer,
    )
    for head in probe.heads.values():
        nn.init.zeros_(head.classifier.weight)
        nn.init.ones_(head.classifier.bias)
    results = run_layerwise_conditions(
        probe,
        [
            {
                "inputs_embeds": torch.zeros(1, probe.embedding_size * 2),
                "labels": torch.tensor([[1.0, 1.0]]),
            }
        ],
        "sequence_multilabel_classification",
        LocalLayerwiseAccelerator(),
        collect_probabilities=True,
    )["trained"]
    for _, predictions, labels, probabilities, _ in results.values():
        assert predictions == labels == [[1, 1]]
        assert probabilities is not None
        assert probabilities[0] == pytest.approx([0.7310586, 0.7310586])


@pytest.mark.parametrize(
    "task_type", ["sequence_classification", "token_classification"]
)
@pytest.mark.parametrize("packed", [False, True])
def test_online_layerwise_batches_share_one_trunk_pass(task_type, packed):
    class PackedTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=1, hidden_size=2)
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def forward(
            self,
            input_ids,
            output_hidden_states=False,
            **kwargs,
        ):
            self.calls += 1
            state = input_ids.unsqueeze(-1).float().expand(-1, -1, 2) * self.weight
            return SimpleNamespace(hidden_states=(state, state + 1))

    trunk = PackedTrunk()
    probe = LayerwiseProbeModel(
        SimpleNamespace(base_model=trunk, config=trunk.config),
        2,
        task_type,
        LayerwiseConfig(enabled=True, layers="all", autocast_dtype="float32"),
        tokenizer=SimpleNamespace(),
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
        "cu_seqlens": torch.tensor([0, 2, 5, 8], dtype=torch.int32),
        "max_seqlen": 3,
        "num_sequences": 2,
        "labels": (
            torch.tensor([0, 1])
            if task_type == "sequence_classification"
            else torch.tensor([[-100, 0, -100, 1, 0, -100, -100, -100]])
        ),
        "id": ["first", "second"],
    }
    if not packed:
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "labels": (
                batch["labels"]
                if task_type == "sequence_classification"
                else torch.tensor([[-100, 0, -100], [-100, 1, 0]])
            ),
            "id": batch["id"],
        }
    with torch.no_grad():
        results = run_layerwise_conditions(
            probe,
            [batch],
            task_type,
            LocalLayerwiseAccelerator(),
            extra_heads={
                "untrained": nn.ModuleDict(
                    {
                        str(layer): EmbeddingProbeHead(2, 2, task_type)
                        for layer in probe.layers
                    }
                )
            },
        )
    assert trunk.calls == 1
    assert set(results) == {"trained", "untrained"}
    for condition in results.values():
        for ids, predictions, labels, _, _ in condition.values():
            assert ids == ["first", "second"]
            assert len(predictions) == len(labels) == 2
            if task_type == "token_classification":
                assert labels == [[0], [1, 0]]


# ---------------------------------------------------------------------------
# pool_hidden_states
# ---------------------------------------------------------------------------


class TestPoolHiddenStates:
    def test_mean_pooling_excludes_special_and_padding_tokens(self):
        tokenizer = SimpleNamespace(cls_token_id=3, eos_token_id=4, pad_token_id=0)
        input_ids = torch.tensor([[3, 5, 6, 4, 0]])
        hidden_states = torch.tensor([[[100.0], [2.0], [4.0], [200.0], [300.0]]])
        mask = residue_attention_mask(input_ids, torch.tensor([[1, 1, 1, 1, 0]]), tokenizer)
        assert pool_hidden_states(hidden_states, mask, "mean").item() == 3.0
        assert pool_hidden_states(hidden_states, mask, "cls").item() == 100.0

    def test_mean_pooling_ignores_padding(self):
        hidden_states = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]]
        )  # (1, 3, 2), last token is padding
        attention_mask = torch.tensor([[True, True, False]])
        pooled = pool_hidden_states(hidden_states, attention_mask, "mean")
        assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))

    def test_cls_pooling_takes_first_token(self):
        hidden_states = torch.tensor([[[5.0, 6.0], [1.0, 1.0]]])
        attention_mask = torch.tensor([[True, True]])
        pooled = pool_hidden_states(hidden_states, attention_mask, "cls")
        assert torch.allclose(pooled, torch.tensor([[5.0, 6.0]]))

    def test_normalization_can_match_v0_before_mean_pooling(self):
        hidden_states = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        attention_mask = torch.tensor([[True, True]])

        pooled = pool_hidden_states(
            hidden_states,
            attention_mask,
            "mean",
            normalize_before_pooling=True,
        )

        assert torch.allclose(pooled, torch.tensor([[0.3, 0.9]]))

    def test_selected_layers_are_concatenated_after_pooling(self):
        hidden_states = (
            torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
            torch.tensor([[[100.0, 200.0], [300.0, 400.0]]]),
        )
        attention_mask = torch.tensor([[True, True]])

        pooled = pool_hidden_state_layers(
            hidden_states, attention_mask, "mean", [0, -1]
        )

        assert torch.allclose(pooled, torch.tensor([[2.0, 3.0, 200.0, 300.0]]))


# ---------------------------------------------------------------------------
# embedding_cache_key
# ---------------------------------------------------------------------------


class TestEmbeddingCacheKey:
    _BASE_KWARGS = dict(
        model_id="flair-bio/amplify-120m",
        dataset_id="biomap-research/metal_ion_binding",
        split="train",
        max_length=512,
        random_truncate=True,
        seed=710019,
        pooling="mean",
        dtype="float16",
        layers=[-1],
    )

    def test_seed_does_not_change_deterministic_split_key(self):
        deterministic_kwargs = {**self._BASE_KWARGS, "random_truncate": False}
        assert embedding_cache_key(
            **{**deterministic_kwargs, "seed": 42}
        ) == embedding_cache_key(**{**deterministic_kwargs, "seed": 710019})

    def test_packed_embeddings_have_separate_cache_key(self):
        assert embedding_cache_key(**self._BASE_KWARGS) != embedding_cache_key(
            **self._BASE_KWARGS, packed=True
        )

    @pytest.mark.parametrize(
        "override",
        [
            {"max_length": 256},
            {"pooling": "cls"},
            {"seed": 42},
            {"layers": [-2]},
            {"input_transform": "scramble_sequences:v1:seed=710019"},
        ],
    )
    def test_changing_any_content_defining_setting_changes_the_key(self, override):
        changed_kwargs = {**self._BASE_KWARGS, **override}
        assert embedding_cache_key(**self._BASE_KWARGS) != embedding_cache_key(
            **changed_kwargs
        )


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class TestEmbeddingCache:
    def test_write_then_read_round_trips(self, tmp_path):
        cache = EmbeddingCache(tmp_path, max_cache_gb=1.0)
        embeddings = torch.randn(4, 8, dtype=torch.float16)
        labels = torch.tensor([0, 1, 0, 1])
        ids = ["a", "b", "c", "d"]

        assert not cache.exists("train", "key1")
        cache.write("train", "key1", embeddings, ids, labels)
        assert cache.exists("train", "key1")
        assert list(tmp_path.iterdir()) == [tmp_path / "train_key1.safetensors"]

        read_embeddings, read_ids, read_labels = cache.read("train", "key1")
        assert torch.equal(read_embeddings, embeddings)
        assert read_ids == ids
        assert torch.equal(read_labels, labels)

    def test_fits_budget_respects_max_cache_gb(self, tmp_path):
        cache = EmbeddingCache(tmp_path, max_cache_gb=1e-6)  # ~1KB budget
        assert not cache.fits_budget(10_000, 960, "float16")
        assert EmbeddingCache(tmp_path, max_cache_gb=1.0).fits_budget(
            10_000, 960, "float16"
        )


# ---------------------------------------------------------------------------
# EmbeddingProbeHead
# ---------------------------------------------------------------------------


class TestEmbeddingProbeHead:
    def test_classification_forward_returns_loss_and_logits(self):
        head = EmbeddingProbeHead(
            hidden_size=8, num_labels=3, task_type="sequence_classification"
        )
        inputs_embeds = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 1])
        output = head(inputs_embeds=inputs_embeds, labels=labels)
        assert output.logits.shape == (4, 3)
        assert output.loss is not None

    def test_mlp_forward_uses_hidden_layer(self):
        head = EmbeddingProbeHead(
            hidden_size=8,
            num_labels=3,
            task_type="sequence_classification",
            probe_hidden_size=4,
        )

        assert isinstance(head.projection, torch.nn.Sequential)
        assert head.classifier.in_features == 4
        output = head(
            inputs_embeds=torch.randn(4, 8), labels=torch.tensor([0, 1, 2, 1])
        )

        assert output.logits.shape == (4, 3)
        assert output.loss is not None

    def test_regression_forward_uses_mse_loss(self):
        head = EmbeddingProbeHead(
            hidden_size=8, num_labels=1, task_type="sequence_regression"
        )
        inputs_embeds = torch.randn(4, 8)
        labels = torch.tensor([0.1, 0.2, -0.3, 1.0])
        output = head(inputs_embeds=inputs_embeds, labels=labels)
        assert output.logits.shape == (4, 1)
        assert output.loss is not None

    def test_multilabel_forward_uses_bce_loss(self):
        head = EmbeddingProbeHead(
            hidden_size=8, num_labels=3, task_type="sequence_multilabel_classification"
        )
        inputs_embeds = torch.randn(4, 8)
        labels = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        output = head(inputs_embeds=inputs_embeds, labels=labels)
        assert output.logits.shape == (4, 3)
        assert output.loss is not None

    def test_forward_without_labels_skips_loss(self):
        head = EmbeddingProbeHead(
            hidden_size=8, num_labels=2, task_type="sequence_classification"
        )
        output = head(inputs_embeds=torch.randn(2, 8))
        assert output.loss is None


class TestExtractEmbeddings:
    def test_packed_sequence_pooling_and_token_selection(self):
        class Trunk(nn.Module):
            def forward(
                self,
                input_ids,
                position_ids,
                cu_seqlens,
                max_seqlen,
                num_sequences,
                output_hidden_states=False,
            ):
                hidden = input_ids.unsqueeze(-1).float()
                return SimpleNamespace(
                    last_hidden_state=hidden, hidden_states=(hidden,)
                )

        class LocalAccelerator:
            device = torch.device("cpu")
            num_processes = 1

            @staticmethod
            def gather_for_metrics(values, **kwargs):
                return values

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]]),
            "position_ids": torch.tensor([[0, 1, 0, 1, 2, 0, 0, 0]]),
            "cu_seqlens": torch.tensor([0, 2, 5, 8], dtype=torch.int32),
            "max_seqlen": 3,
            "num_sequences": 2,
            "labels": torch.tensor([0, 1]),
            "id": ["first", "second"],
        }
        pooled, ids, labels, _ = extract_embeddings(
            Trunk(), [batch], LocalAccelerator(), "mean", "float32", layers=[-1]
        )
        assert pooled[:, 0].tolist() == [1.5, 4.0]
        assert ids == ["first", "second"]
        assert labels.tolist() == [0, 1]

        token_batch = dict(
            batch, labels=torch.tensor([[-100, 2, -100, 3, 4, -100, -100, -100]])
        )
        tokens, ids, labels, _ = extract_embeddings(
            Trunk(),
            [token_batch],
            LocalAccelerator(),
            "mean",
            "float32",
            layers=[-1],
            is_ragged=True,
        )
        assert tokens[:, 0].tolist() == [2.0, 4.0, 5.0]
        assert ids == ["first", "second", "second"]
        assert labels.tolist() == [2, 3, 4]

    def test_uses_base_model_without_running_task_head(self):
        class Trunk(nn.Module):
            def forward(self, input_ids, attention_mask, output_hidden_states=False):
                assert output_hidden_states is False
                hidden = input_ids.unsqueeze(-1).float()
                return SimpleNamespace(
                    last_hidden_state=hidden + 1,
                    hidden_states=(hidden, hidden + 1),
                )

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.base_model = Trunk()

            def forward(self, **kwargs):
                raise AssertionError("The task head must not run during extraction.")

        class LocalAccelerator:
            @staticmethod
            def gather_for_metrics(values, **kwargs):
                if kwargs.get("use_gather_object"):
                    return values + ["padding-id"]
                return values

        dataloader = [
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[True, True]]),
                "labels": torch.tensor([1]),
                "id": ["example-1"],
            }
        ]

        embeddings, ids, labels, _ = extract_embeddings(
            Wrapper(), dataloader, LocalAccelerator(), "mean", "float32", layers=[-1]
        )

        assert torch.equal(embeddings, torch.tensor([[2.5]]))
        assert ids == ["example-1"]
        assert torch.equal(labels, torch.tensor([1]))


# ---------------------------------------------------------------------------
# PrepareStep integration (embedding_cache enabled)
# ---------------------------------------------------------------------------


class TestPrepareStepWithEmbeddingCache:
    def test_layerwise_reuses_multilayer_sequence_cache(
        self, monkeypatch, tmp_path, model_family
    ):
        from modules.evaluate.src import utils

        calls = []
        original_extract = utils.embed.extract_embeddings

        def count_extract(*args, **kwargs):
            calls.append(kwargs["layers"])
            return original_extract(*args, **kwargs)

        monkeypatch.setattr(utils.embed, "extract_embeddings", count_extract)
        config = LayerwiseConfig(enabled=True, layers=[0, -1], autocast_dtype="float32")
        cache = EmbeddingCacheConfig(enabled=True, dtype="float32")
        torch.manual_seed(123)
        _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            layerwise_config=config,
            embedding_cache=cache,
        )
        assert calls == [[0, 2]] * 3
        calls.clear()
        torch.manual_seed(123)
        _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            layerwise_config=config,
            embedding_cache=cache,
        )
        assert calls == []

    def test_swaps_in_probe_head_and_embedding_batches(
        self, monkeypatch, tmp_path, model_family
    ):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )

        assert isinstance(artifacts.model, EmbeddingProbeHead)

        for dataloader in (
            artifacts.train_dataloader,
            artifacts.val_dataloader,
            artifacts.test_dataloader,
        ):
            batch = next(iter(dataloader))
            assert "inputs_embeds" in batch
            assert "input_ids" not in batch

            output = artifacts.model(**{k: v for k, v in batch.items() if k != "id"})
            assert output.loss is not None

    def test_all_sequence_splits_are_disk_cached(
        self, monkeypatch, tmp_path, model_family
    ):
        _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )

        workspace = get_evaluation_workspace(
            EvaluationWorkspaceConfig(
                dataset_name="toy_dataset",
                dataset_repo_id="biomap-research/toy_dataset",
                model_name="toy-model",
                model_repo_id="flair-bio/toy-model",
                task_type="sequence_classification",
                target_type="categorical",
                label_format="scalar",
                num_labels=2,
                base_path=str(tmp_path / "eval"),
            )
        )

        cached_files = list(workspace.embeddings_path.glob("*.safetensors"))
        cached_splits = {path.stem.split("_")[0] for path in cached_files}
        assert cached_splits == {"train", "validation", "test"}

    def test_selected_layers_expand_the_probe_input_width(
        self, monkeypatch, tmp_path, model_family
    ):
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "sequence_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True, layers=[-1, -2]),
        )

        assert artifacts.embedding_backend is not None
        assert (
            artifacts.embedding_backend.hidden_size
            == model_family.config_kwargs["hidden_size"] * 2
        )
        assert artifacts.embedding_backend.layers == [-1, -2]
        batch = next(iter(artifacts.train_dataloader))
        assert (
            batch["inputs_embeds"].shape[-1] == artifacts.embedding_backend.hidden_size
        )

    def test_token_classification_uses_unpooled_per_token_embeddings(
        self, monkeypatch, tmp_path, model_family
    ):
        """Unlike sequence-level tasks, token_classification keeps one row
        per valid token (not per example), and is never disk-cached."""
        artifacts = _run_prepare(
            monkeypatch,
            tmp_path,
            "token_classification",
            model_family,
            embedding_cache=EmbeddingCacheConfig(enabled=True),
        )

        assert isinstance(artifacts.model, EmbeddingProbeHead)
        # 4 examples per split (see build_raw_dataset_dict); more rows than
        # examples confirms rows are per-token, not per-example.
        assert len(artifacts.train_dataloader.dataset) > 4

        batch = next(iter(artifacts.train_dataloader))
        assert "inputs_embeds" in batch
        assert batch["labels"].shape == (batch["inputs_embeds"].shape[0],)
        output = artifacts.model(**{k: v for k, v in batch.items() if k != "id"})
        assert output.loss is not None

        workspace = get_evaluation_workspace(
            EvaluationWorkspaceConfig(
                dataset_name="toy_dataset",
                dataset_repo_id="biomap-research/toy_dataset",
                model_name="toy-model",
                model_repo_id="flair-bio/toy-model",
                task_type="token_classification",
                target_type="categorical",
                label_format="per_token",
                num_labels=2,
                base_path=str(tmp_path / "eval"),
            )
        )
        assert list(workspace.embeddings_path.glob("*.safetensors")) == []


# ---------------------------------------------------------------------------
# EvaluationConfig cross-step validation
# ---------------------------------------------------------------------------


class TestEmbeddingCacheCrossStepValidation:
    def test_rejects_embedding_cache_outside_tune_probe(self):
        with pytest.raises(Exception, match="mode='finetune'.*embedding_cache"):
            EvaluationConfig(
                seed=1,
                workspace=dict(
                    dataset_name="d",
                    model_name="m",
                    task_type="sequence_classification",
                    target_type="categorical",
                    label_format="scalar",
                    num_labels=2,
                ),
                steps=dict(
                    mode="finetune",
                    prepare=dict(embedding_cache=dict(enabled=True)),
                ),
            )

    def test_allows_embedding_cache_with_tune_probe(self):
        config = EvaluationConfig(
            seed=1,
            workspace=dict(
                dataset_name="d",
                model_name="m",
                task_type="sequence_classification",
                target_type="categorical",
                label_format="scalar",
                num_labels=2,
            ),
            steps=dict(
                mode="tune_probe",
                prepare=dict(embedding_cache=dict(enabled=True)),
            ),
        )
        assert config.steps.prepare.embedding_cache.enabled is True
