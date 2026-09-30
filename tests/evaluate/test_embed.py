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

from modules.evaluate.src.config import EvaluationConfig
from modules.evaluate.src.dataset.workspace import (
    EvaluationWorkspaceConfig,
    get_evaluation_workspace,
)
from modules.evaluate.src.utils.embed import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    EmbeddingDataset,
    EmbeddingProbeHead,
    embedding_collate_fn,
    extract_embeddings,
    embedding_cache_key,
    estimate_embedding_bytes,
    pool_hidden_state_layers,
    pool_hidden_states,
)

from .test_prepare import _run_prepare

import pytest

# ---------------------------------------------------------------------------
# pool_hidden_states
# ---------------------------------------------------------------------------


class TestPoolHiddenStates:
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
# estimate_embedding_bytes
# ---------------------------------------------------------------------------


class TestEstimateEmbeddingBytes:
    def test_matches_itemsize_arithmetic(self):
        assert estimate_embedding_bytes(1000, 960, "float16") == 1000 * 960 * 2
        assert estimate_embedding_bytes(1000, 960, "float32") == 1000 * 960 * 4


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
    def test_is_a_single_pytorch_linear_layer(self):
        head = EmbeddingProbeHead(
            hidden_size=8, num_labels=3, task_type="sequence_classification"
        )

        assert isinstance(head.classifier, torch.nn.Linear)

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
