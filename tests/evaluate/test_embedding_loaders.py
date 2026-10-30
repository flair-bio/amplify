"""Ordering and cache regressions for frozen-trunk extraction loaders."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from torch import nn

from modules.evaluate.src.artifacts.feature_cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
)
from modules.evaluate.src.config.predict import PredictConfig
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import DataLoaderConfig, build_dataloaders
from modules.evaluate.src.model.heads import HeadConfig
from modules.evaluate.src.predictors.embeddings import (
    build_embedding_dataloaders,
    extract_embeddings,
    find_embedding_dataset,
)
from modules.evaluate.src.predictors.inference import predict_trained
from modules.evaluate.src.training.config import TuneConfig
from modules.evaluate.src.training.torch_trainer import TorchProbeTrainer
from modules.evaluate.src.training.tuning import fit_probe_protocol

from .test_prepare import _make_workspace, _run_prepare


class _RecordingTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.padded_tokens = 0
        self.batch_sizes = []

    def forward(self, input_ids, **kwargs):
        self.padded_tokens += input_ids.numel()
        self.batch_sizes.append(kwargs.get("num_sequences", input_ids.shape[0]))
        values = input_ids.float()
        states = torch.stack((values, values.square()), dim=-1) * self.weight
        return SimpleNamespace(last_hidden_state=states, hidden_states=(states,))


def _token_loaders(task_type, grouped, packed=False):
    lengths = [3, 18, 5, 18, 4]
    tokens = [
        [99, *range(5 + index * 20, 5 + index * 20 + length - 2), 98]
        for index, length in enumerate(lengths)
    ]
    labels = (
        [
            [-100, *[position % 2 for position in range(length - 2)], -100]
            for length in lengths
        ]
        if task_type == "token_classification"
        else [index % 2 for index in range(len(lengths))]
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": tokens,
            "attention_mask": [[1] * length for length in lengths],
            "targets": labels,
            "id": [f"example_{index}" for index in range(len(lengths))],
        }
    )
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=98, cls_token_id=99)
    collator = EvaluationCollator(
        tokenizer,
        task_type,
        EvaluationCollatorConfig(pad_to_multiple_of=None, packed=packed),
    )
    loaders = build_dataloaders(
        DatasetDict({split: dataset for split in ("train", "validation", "test")}),
        collator,
        DataLoaderConfig(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            length_grouped_sampling=grouped,
            max_tokens_per_batch=24 if packed else None,
        ),
    )
    return dataset, loaders


def _embedding_loaders(
    model,
    loaders,
    accelerator,
    tmp_path,
    grouped,
    task_type,
    extraction_batch_size=3,
    cache_policy="off",
):
    return build_embedding_dataloaders(
        model=model,
        splits={
            split: (loader, False)
            for split, loader in zip(("train", "validation", "test"), loaders)
        },
        accelerator=accelerator,
        cache=EmbeddingCache(tmp_path / "cache", 1.0),
        cache_config=EmbeddingCacheConfig(
            enabled=True,
            dtype="float32",
            normalize_before_pooling=False,
            cache_policy=cache_policy,
        ),
        model_id="toy-model",
        dataset_id="toy-data",
        max_length=64,
        seed=17,
        batch_size=2,
        embedding_size=2,
        label_dtype=torch.long,
        extraction_batch_size=extraction_batch_size,
        is_ragged=task_type == "token_classification",
        length_grouped_sampling=grouped,
    )


@pytest.mark.parametrize(
    "task_type", ["sequence_classification", "token_classification"]
)
@pytest.mark.parametrize("extraction_batch_size", [2, 3])
def test_grouped_extraction_preserves_order_features_and_controls(
    tmp_path, task_type, extraction_batch_size
):
    accelerator = Accelerator(cpu=True)
    reference_splits = None
    reference_padding = None
    for grouped in (False, True):
        dataset, raw_loaders = _token_loaders(task_type, grouped)
        model = _RecordingTrunk()
        cached_loaders = _embedding_loaders(
            model,
            accelerator.prepare(*raw_loaders),
            accelerator,
            tmp_path,
            grouped,
            task_type,
            extraction_batch_size=extraction_batch_size,
        )
        padding = model.padded_tokens
        indices = (
            sorted(
                range(len(dataset)),
                key=lambda index: len(dataset[index]["input_ids"]),
                reverse=True,
            )
            if grouped
            else list(range(len(dataset)))
        )
        expected_ids = []
        expected_labels = []
        for index in indices:
            row = dataset[index]
            labels = (
                [label for label in row["targets"] if label != -100]
                if task_type == "token_classification"
                else [row["targets"]]
            )
            expected_ids.extend([row["id"]] * len(labels))
            expected_labels.extend(labels)
        splits = {}
        for split, loader in cached_loaders.items():
            cached = find_embedding_dataset(loader)
            assert cached is not None
            assert cached.ids == expected_ids
            assert cached.labels.tolist() == expected_labels
            splits[split] = cached
            if reference_splits is not None:
                reference = reference_splits[split]
                for identifier in set(expected_ids):
                    current_indices = [
                        i for i, value in enumerate(cached.ids) if value == identifier
                    ]
                    reference_indices = [
                        i
                        for i, value in enumerate(reference.ids)
                        if value == identifier
                    ]
                    torch.testing.assert_close(
                        cached.embeddings[current_indices],
                        reference.embeddings[reference_indices],
                    )
        control_embeddings, control_ids, control_labels, _ = extract_embeddings(
            model,
            raw_loaders[2],
            accelerator,
            "mean",
            "float32",
            is_ragged=task_type == "token_classification",
            gather_across_processes=False,
        )
        assert control_ids == splits["test"].ids
        assert torch.equal(control_labels, splits["test"].labels)
        torch.testing.assert_close(control_embeddings, splits["test"].embeddings)
        train_orders = [
            [
                identifier
                for batch in cached_loaders["train"]
                for identifier in batch["id"]
            ]
            for _ in range(2)
        ]
        assert train_orders[0] != train_orders[1]
        assert sorted(train_orders[0]) == sorted(expected_ids)
        assert sorted(train_orders[1]) == sorted(expected_ids)
        if grouped:
            assert padding < reference_padding
        else:
            reference_splits = splits
            reference_padding = padding


@pytest.mark.parametrize("grouped", [False, True])
def test_extraction_order_does_not_depend_on_batch_size(tmp_path, grouped):
    accelerator = Accelerator(cpu=True)
    _, raw_loaders = _token_loaders("sequence_classification", grouped)
    orders = []
    for batch_size in (2, 3, 128):
        loaders = _embedding_loaders(
            _RecordingTrunk(),
            accelerator.prepare(*raw_loaders),
            accelerator,
            tmp_path,
            grouped,
            "sequence_classification",
            extraction_batch_size=batch_size,
        )
        orders.append(find_embedding_dataset(loaders["test"]).ids)
    assert orders[0] == orders[1] == orders[2]


def test_grouping_has_separate_cache_entries_and_stable_warm_order(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = _RecordingTrunk()
    for grouped in (False, True):
        _, raw_loaders = _token_loaders("sequence_classification", grouped)
        prepared_loaders = accelerator.prepare(*raw_loaders)
        cold = _embedding_loaders(
            model,
            prepared_loaders,
            accelerator,
            tmp_path,
            grouped,
            "sequence_classification",
            cache_policy="reuse",
        )
        assert model.padded_tokens > 0
        model.padded_tokens = 0
        warm = _embedding_loaders(
            model,
            prepared_loaders,
            accelerator,
            tmp_path,
            grouped,
            "sequence_classification",
            cache_policy="reuse",
        )
        assert model.padded_tokens == 0
        for split in cold:
            cold_split = find_embedding_dataset(cold[split])
            warm_split = find_embedding_dataset(warm[split])
            assert cold_split.ids == warm_split.ids
            assert torch.equal(cold_split.labels, warm_split.labels)
            assert torch.equal(cold_split.embeddings, warm_split.embeddings)
        assert len(list((tmp_path / "cache").glob("*.safetensors"))) == (
            6 if grouped else 3
        )


def test_packed_extraction_keeps_token_budget_batches(tmp_path):
    accelerator = Accelerator(cpu=True)
    dataset, raw_loaders = _token_loaders("sequence_classification", True, packed=True)
    expected_batches = [batch["num_sequences"] for batch in raw_loaders[2]]
    model = _RecordingTrunk()
    loaders = _embedding_loaders(
        model,
        (None, None, accelerator.prepare(raw_loaders[2])),
        accelerator,
        tmp_path,
        True,
        "sequence_classification",
        extraction_batch_size=None,
    )
    assert model.batch_sizes == expected_batches
    assert find_embedding_dataset(loaders["test"]).ids == dataset["id"][:]


@pytest.mark.parametrize(
    "task_type", ["sequence_classification", "token_classification"]
)
@pytest.mark.parametrize("grouped", [False, True])
def test_prepare_and_prediction_controls_use_matching_order(
    monkeypatch, tmp_path, model_family, task_type, grouped
):
    artifacts = _run_prepare(
        monkeypatch,
        tmp_path,
        task_type,
        model_family,
        tokenize_num_proc=1,
        length_grouped_sampling=grouped,
        embedding_batch_size=3,
        embedding_cache=EmbeddingCacheConfig(enabled=True, dtype="float32"),
    )
    backend = artifacts.embedding_backend
    assert backend.length_grouped_sampling == grouped
    other_grouping = replace(backend, length_grouped_sampling=not grouped)
    for transform in (None, "scramble_sequences:v1:seed=710019"):
        assert backend.test_embedding_key(
            trunk=backend.trunk, seed=710019, input_transform=transform
        ) != other_grouping.test_embedding_key(
            trunk=backend.trunk, seed=710019, input_transform=transform
        )
    if task_type == "sequence_classification":
        assert backend.cache.exists("test", backend.test_cache_key)
    controls = ["random_trunk", "random_both"]
    if task_type == "sequence_classification":
        controls.append("scrambled_sequences")
    fit_probe_protocol(
        TorchProbeTrainer(
            TuneConfig(
                head=HeadConfig(head_type="torch_linear"),
                max_epochs=1,
                save_best_model=False,
            )
        ),
        _make_workspace(tmp_path, task_type),
        artifacts,
        controls=controls,
    )
    output = predict_trained(
        _make_workspace(tmp_path, task_type),
        artifacts,
        PredictConfig(controls=controls),
    )
    for control in controls:
        assert (
            output.controls[control]["id"].to_list()
            == output.predictions["id"].to_list()
        )
