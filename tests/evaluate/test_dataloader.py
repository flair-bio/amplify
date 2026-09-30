"""Tests for modules.evaluate.src.dataset.dataloader.

Uses the real ``ProteinTokenizer`` and small in-memory HF ``Dataset``/
``DatasetDict`` objects (see tests/evaluate/conftest.py) to exercise
tokenization, truncation, and DataLoader construction end-to-end without
any network access.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset, DatasetDict

from modules.evaluate.src.dataset import dataloader as dataloader_module
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    DatasetSplitConfig,
    LengthGroupedSampler,
    TokenBudgetBatchSampler,
    build_dataloaders,
    default_num_workers,
    ensure_validation_split,
    load_evaluation_dataset,
    select_dataset_splits,
    tokenize_dataset,
)
from modules.evaluate.src.utils.seed import seed_everything

from .conftest import LONG_SEQUENCE, SHORT_SEQUENCE, build_raw_dataset_dict


def test_token_budget_sampler_keeps_each_example_once():
    lengths = [4, 5, 3, 14, 2, 6]
    sampler = TokenBudgetBatchSampler(
        lengths, 9, True, torch.Generator().manual_seed(17)
    )
    first = list(sampler)
    second = list(sampler)
    assert len(first) == len(second) == len(sampler)
    assert sorted(index for batch in first for index in batch) == list(
        range(len(lengths))
    )
    assert all(
        sum(lengths[index] for index in batch) <= 9 or len(batch) == 1
        for batch in first
    )
    assert first != second


# ---------------------------------------------------------------------------
# default_num_workers
# ---------------------------------------------------------------------------


class TestDefaultNumWorkers:
    @pytest.mark.parametrize(
        ("gpu_count", "cpu_count", "expected"),
        [(0, 8, 4), (0, 2, 2), (8, 4, 4), (1, 64, 4)],
    )
    def test_uses_the_smaller_available_worker_budget(
        self, monkeypatch, gpu_count, cpu_count, expected
    ):
        monkeypatch.setattr(
            dataloader_module.torch.cuda, "device_count", lambda: gpu_count
        )
        monkeypatch.setattr(dataloader_module.os, "cpu_count", lambda: cpu_count)
        assert default_num_workers() == expected


# ---------------------------------------------------------------------------
# load_evaluation_dataset
# ---------------------------------------------------------------------------


class TestLoadEvaluationDataset:
    def test_returns_dataset_dict(self, monkeypatch):
        raw = build_raw_dataset_dict("sequence_classification")
        monkeypatch.setattr(dataloader_module, "load_dataset", lambda repo_id: raw)
        result = load_evaluation_dataset("some/repo")
        assert result is raw

    def test_raises_when_not_a_dataset_dict(self, monkeypatch):
        single_split = Dataset.from_dict({"sequence": [SHORT_SEQUENCE], "targets": [0]})
        monkeypatch.setattr(
            dataloader_module, "load_dataset", lambda repo_id: single_split
        )
        with pytest.raises(ValueError, match="DatasetDict"):
            load_evaluation_dataset("some/repo")

    def test_synthesizes_validation_split_when_missing(self, monkeypatch):
        raw = build_raw_dataset_dict(
            "sequence_classification", splits=("train", "test")
        )
        monkeypatch.setattr(dataloader_module, "load_dataset", lambda repo_id: raw)
        result = load_evaluation_dataset("some/repo", seed=0, validation_fraction=0.5)
        assert set(result.keys()) == {"train", "validation", "test"}
        assert len(result["train"]) + len(result["validation"]) == len(raw["train"])
        assert result["test"] is raw["test"]


class TestSelectDatasetSplits:
    def test_maps_named_source_splits_to_pipeline_roles(self):
        raw = build_raw_dataset_dict(
            "sequence_classification",
            splits=(
                "train_low_mutation",
                "validation_low_mutation",
                "test_high_mutation",
            ),
        )

        result = select_dataset_splits(
            raw,
            DatasetSplitConfig(
                name="low_to_high_mutation",
                train="train_low_mutation",
                validation="validation_low_mutation",
                test="test_high_mutation",
            ),
        )

        assert set(result) == {"train", "validation", "test"}
        assert result["train"] is raw["train_low_mutation"]
        assert result["validation"] is raw["validation_low_mutation"]
        assert result["test"] is raw["test_high_mutation"]

    def test_rejects_missing_configured_source_split(self):
        raw = build_raw_dataset_dict("sequence_classification")
        with pytest.raises(ValueError, match="test_at_30_identity"):
            select_dataset_splits(
                raw,
                DatasetSplitConfig(
                    name="identity_30",
                    train="train",
                    validation="validation",
                    test="test_at_30_identity",
                ),
            )

    @pytest.mark.parametrize("method", ["random", "stratified", "cluster"])
    def test_derives_suffixed_split_names_from_split_name(self, method):
        config = DatasetSplitConfig(name=method)
        assert (config.train, config.validation, config.test) == (
            f"train_{method}",
            f"validation_{method}",
            f"test_{method}",
        )

    def test_default_name_keeps_plain_split_names(self):
        config = DatasetSplitConfig()
        assert (config.train, config.validation, config.test) == (
            "train",
            "validation",
            "test",
        )

    def test_explicit_role_overrides_derived_name(self):
        config = DatasetSplitConfig(name="low_to_high", test="test_high_mutation")
        assert config.train == "train_low_to_high"
        assert config.test == "test_high_mutation"

    def test_raises_when_derived_split_is_missing_from_dataset(self):
        raw = build_raw_dataset_dict("sequence_classification")
        with pytest.raises(ValueError, match="train_stratified"):
            select_dataset_splits(raw, DatasetSplitConfig(name="stratified"))


# ---------------------------------------------------------------------------
# ensure_validation_split
# ---------------------------------------------------------------------------


class TestEnsureValidationSplit:
    def test_leaves_dataset_unchanged_when_validation_already_present(self):
        raw = build_raw_dataset_dict("sequence_classification")
        result = ensure_validation_split(raw)
        assert result is raw

    def test_leaves_dataset_unchanged_when_no_train_split(self):
        raw = build_raw_dataset_dict("sequence_classification", splits=("test",))
        result = ensure_validation_split(raw)
        assert result is raw

    def test_splits_train_when_validation_missing(self, caplog):
        raw = build_raw_dataset_dict(
            "sequence_classification", splits=("train", "test")
        )
        with caplog.at_level("WARNING"):
            result = ensure_validation_split(raw, seed=0, validation_fraction=0.5)
        assert set(result.keys()) == {"train", "validation", "test"}
        assert len(result["train"]) + len(result["validation"]) == len(raw["train"])
        assert len(result["validation"]) > 0
        assert any("synthesized" in record.message.lower() for record in caplog.records)

    def test_deterministic_given_seed(self):
        raw = build_raw_dataset_dict(
            "sequence_classification", splits=("train", "test")
        )
        result_a = ensure_validation_split(raw, seed=1, validation_fraction=0.5)
        result_b = ensure_validation_split(raw, seed=1, validation_fraction=0.5)
        assert result_a["validation"]["sequence"] == result_b["validation"]["sequence"]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# tokenize_dataset
# ---------------------------------------------------------------------------


class TestTokenizeDatasetSequenceTasks:
    def test_rejects_max_length_without_room_for_residues(self, protein_tokenizer):
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        with pytest.raises(ValueError, match="must exceed"):
            tokenize_dataset(
                raw,
                tokenizer=protein_tokenizer,
                task_type="sequence_classification",
                sequence_column="sequence",
                label_column="targets",
                max_length=2,
            )

    def test_random_truncate_path_respects_max_length(self, protein_tokenizer):
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        max_length = 20
        tokenized = tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=max_length,
            num_proc=None,
            random_truncate=True,
        )
        for example in tokenized["train"]:
            assert len(example["input_ids"]) <= max_length
            # BOS/EOS preserved at the boundaries.
            assert example["input_ids"][0] == protein_tokenizer.bos_token_id
            assert example["input_ids"][-1] == protein_tokenizer.eos_token_id
            assert len(example["attention_mask"]) == len(example["input_ids"])

    def test_short_sequences_untruncated(self, protein_tokenizer):
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        tokenized = tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )
        # SHORT_SEQUENCE (index 0) is well under max_length: BOS + residues + EOS.
        example = tokenized["train"][0]
        assert len(example["input_ids"]) == len(SHORT_SEQUENCE) + 2

    def test_deterministic_truncation_uses_seed_everything(self, protein_tokenizer):
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        kwargs = dict(
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=15,
            num_proc=None,
            random_truncate=True,
        )
        seed_everything(123)
        first = tokenize_dataset(raw, seed=123, **kwargs)["train"]["input_ids"]
        seed_everything(123)
        second = tokenize_dataset(raw, seed=123, **kwargs)["train"]["input_ids"]
        assert first == second

        third = tokenize_dataset(raw, seed=456, **kwargs)["train"]["input_ids"]
        assert first != third


class TestTokenizeDatasetSequenceTasks:
    @pytest.mark.parametrize(
        "tokenizer_fixture", ["protein_tokenizer", "esm2_tokenizer"]
    )
    def test_random_truncate_path_respects_max_length(self, request, tokenizer_fixture):
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        max_length = 20
        tokenized = tokenize_dataset(
            raw,
            tokenizer=tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=max_length,
            num_proc=None,
            random_truncate=True,
        )
        for example in tokenized["train"]:
            assert len(example["input_ids"]) <= max_length
            assert example["input_ids"][0] in {
                tokenizer.bos_token_id,
                tokenizer.cls_token_id,
            }
            assert example["input_ids"][-1] == tokenizer.eos_token_id
            assert len(example["attention_mask"]) == len(example["input_ids"])

    @pytest.mark.parametrize(
        "tokenizer_fixture", ["protein_tokenizer", "esm2_tokenizer"]
    )
    def test_short_sequences_untruncated(self, request, tokenizer_fixture):
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        raw = build_raw_dataset_dict("sequence_classification", splits=("train",))
        tokenized = tokenize_dataset(
            raw,
            tokenizer=tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )
        example = tokenized["train"][0]
        assert len(example["input_ids"]) == len(SHORT_SEQUENCE) + 2


class TestTokenizeDatasetTokenClassification:
    @pytest.mark.parametrize("labels", [[0, 1], [0] * (len(SHORT_SEQUENCE) + 1)])
    def test_rejects_labels_that_do_not_match_residue_count(
        self, protein_tokenizer, labels
    ):
        raw = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"sequence": [SHORT_SEQUENCE], "targets": [labels]}
                )
            }
        )
        with pytest.raises(ValueError, match="one label per tokenized residue"):
            tokenize_dataset(
                raw,
                tokenizer=protein_tokenizer,
                task_type="token_classification",
                sequence_column="sequence",
                label_column="targets",
                max_length=512,
                num_proc=None,
            )

    def test_labels_aligned_with_bos_eos(self, protein_tokenizer):
        raw = build_raw_dataset_dict("token_classification", splits=("train",))
        tokenized = tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )
        example = tokenized["train"][0]
        # Length matches input_ids exactly (label per token, incl. BOS/EOS).
        assert len(example["targets"]) == len(example["input_ids"])
        assert example["targets"][0] == -100
        assert example["targets"][-1] == -100
        # Interior residue labels are preserved (not -100).
        assert all(label != -100 for label in example["targets"][1:-1])

    def test_labels_aligned_with_bos_eos_esm2(self, esm2_tokenizer):
        """Same alignment guarantee holds for the ESM2 (cls/eos) tokenizer."""
        raw = build_raw_dataset_dict("token_classification", splits=("train",))
        tokenized = tokenize_dataset(
            raw,
            tokenizer=esm2_tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )
        example = tokenized["train"][0]
        assert len(example["targets"]) == len(example["input_ids"])
        assert example["input_ids"][0] == esm2_tokenizer.cls_token_id
        assert example["input_ids"][-1] == esm2_tokenizer.eos_token_id
        assert example["targets"][0] == -100
        assert example["targets"][-1] == -100
        assert all(label != -100 for label in example["targets"][1:-1])

    def test_manual_truncation_slices_labels_consistently(self, protein_tokenizer):
        raw = build_raw_dataset_dict("token_classification", splits=("train",))
        max_length = 10  # forces truncation for LONG_SEQUENCE (index 1)
        tokenized = tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=max_length,
            num_proc=None,
            random_truncate=False,  # deterministic left-window for easy assertion
        )
        example = tokenized["train"][1]  # LONG_SEQUENCE
        assert len(example["input_ids"]) <= max_length
        assert len(example["targets"]) == len(example["input_ids"])
        assert example["input_ids"][0] == protein_tokenizer.bos_token_id
        assert example["input_ids"][-1] == protein_tokenizer.eos_token_id
        assert example["targets"][0] == -100
        assert example["targets"][-1] == -100

    def test_manual_truncation_slices_labels_consistently_esm2(self, esm2_tokenizer):
        """ESM2 always takes the manual-crop path (no native random_truncate)."""
        raw = build_raw_dataset_dict("token_classification", splits=("train",))
        max_length = 10
        tokenized = tokenize_dataset(
            raw,
            tokenizer=esm2_tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=max_length,
            num_proc=None,
            random_truncate=False,
        )
        example = tokenized["train"][1]  # LONG_SEQUENCE
        assert len(example["input_ids"]) <= max_length
        assert len(example["targets"]) == len(example["input_ids"])
        assert example["input_ids"][0] == esm2_tokenizer.cls_token_id
        assert example["input_ids"][-1] == esm2_tokenizer.eos_token_id
        assert example["targets"][0] == -100
        assert example["targets"][-1] == -100


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------


class TestBuildDataloaders:
    def _tokenized(self, protein_tokenizer, splits=("train", "validation")):
        raw = build_raw_dataset_dict("sequence_classification", splits=splits)
        return tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )

    def test_missing_splits_yield_none(self, protein_tokenizer):
        tokenized = self._tokenized(protein_tokenizer, splits=("train",))
        collator = EvaluationCollator(
            protein_tokenizer,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets"),
        )
        train_dl, val_dl, test_dl = build_dataloaders(
            tokenized, collator, DataLoaderConfig(batch_size=2, num_workers=0)
        )
        assert train_dl is not None
        assert val_dl is None
        assert test_dl is None

    def test_batches_have_expected_shapes(self, protein_tokenizer):
        tokenized = self._tokenized(protein_tokenizer)
        collator = EvaluationCollator(
            protein_tokenizer,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets", pad_to_multiple_of=8),
        )
        train_dl, val_dl, test_dl = build_dataloaders(
            tokenized,
            collator,
            DataLoaderConfig(
                batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False
            ),
        )

        assert val_dl is not None
        assert test_dl is None

        batch = next(iter(train_dl))
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].dtype == torch.long
        assert batch["labels"].shape == (2,)

        total_examples = sum(b["input_ids"].shape[0] for b in train_dl)
        assert total_examples == len(tokenized["train"])

    def test_batches_have_expected_shapes_esm2(self, esm2_tokenizer):
        """Same DataLoader construction works with the real ESM2 tokenizer/pad id."""
        raw = build_raw_dataset_dict("sequence_classification")
        tokenized = tokenize_dataset(
            raw,
            tokenizer=esm2_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )
        collator = EvaluationCollator(
            esm2_tokenizer,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets", pad_to_multiple_of=8),
        )
        train_dl, val_dl, test_dl = build_dataloaders(
            tokenized,
            collator,
            DataLoaderConfig(
                batch_size=2, num_workers=0, pin_memory=False, persistent_workers=False
            ),
        )

        batch = next(iter(train_dl))
        assert batch["input_ids"].shape[0] == 2
        assert torch.all(
            (batch["input_ids"] != esm2_tokenizer.pad_token_id)
            == batch["attention_mask"]
        )


# ---------------------------------------------------------------------------
# LengthGroupedSampler / length_grouped_sampling
# ---------------------------------------------------------------------------


class TestLengthGroupedSampler:
    def test_yields_all_indices_exactly_once(self):
        lengths = [5, 50, 6, 48, 4, 52, 7, 49, 3, 51]
        sampler = LengthGroupedSampler(lengths, batch_size=2)
        indices = list(iter(sampler))
        assert sorted(indices) == list(range(len(lengths)))

    def test_len_matches_number_of_lengths(self):
        lengths = [1, 2, 3, 4]
        sampler = LengthGroupedSampler(lengths, batch_size=2)
        assert len(sampler) == 4

    def test_consecutive_batches_have_similar_lengths(self):
        # Two clearly separated length clusters; batches should not mix them.
        lengths = [5, 6, 7, 8] + [100, 101, 102, 103]
        sampler = LengthGroupedSampler(
            lengths, batch_size=4, generator=torch.Generator().manual_seed(0)
        )
        indices = list(iter(sampler))
        for i in range(0, len(indices), 4):
            batch_lengths = [lengths[j] for j in indices[i : i + 4]]
            assert max(batch_lengths) - min(batch_lengths) < 10

    def test_generator_makes_order_reproducible(self):
        lengths = [5, 50, 6, 48, 4, 52, 7, 49, 3, 51]
        first = list(
            iter(LengthGroupedSampler(lengths, 2, torch.Generator().manual_seed(42)))
        )
        second = list(
            iter(LengthGroupedSampler(lengths, 2, torch.Generator().manual_seed(42)))
        )
        assert first == second


class TestBuildDataloadersLengthGrouped:
    def _tokenized(self, protein_tokenizer, splits=("train", "validation")):
        raw = build_raw_dataset_dict("sequence_classification", splits=splits)
        return tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=512,
            num_proc=None,
        )

    def test_length_grouped_sampling_covers_all_examples(self, protein_tokenizer):
        tokenized = self._tokenized(protein_tokenizer)
        collator = EvaluationCollator(
            protein_tokenizer,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets"),
        )
        train_dl, val_dl, test_dl = build_dataloaders(
            tokenized,
            collator,
            DataLoaderConfig(
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                length_grouped_sampling=True,
            ),
        )
        assert val_dl is not None
        total_examples = sum(b["input_ids"].shape[0] for b in train_dl)
        assert total_examples == len(tokenized["train"])
        total_val_examples = sum(b["input_ids"].shape[0] for b in val_dl)
        assert total_val_examples == len(tokenized["validation"])

    def test_length_grouped_sampling_reduces_padding(self, protein_tokenizer):
        tokenized = self._tokenized(protein_tokenizer, splits=("train",))
        collator = EvaluationCollator(
            protein_tokenizer,
            "sequence_classification",
            EvaluationCollatorConfig(label_column="targets", pad_to_multiple_of=None),
        )

        def total_padded_tokens(length_grouped: bool) -> int:
            train_dl, _, _ = build_dataloaders(
                tokenized,
                collator,
                DataLoaderConfig(
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False,
                    persistent_workers=False,
                    length_grouped_sampling=length_grouped,
                    seed=0,
                ),
            )
            return sum(b["input_ids"].numel() for b in train_dl)

        assert total_padded_tokens(True) <= total_padded_tokens(False)
