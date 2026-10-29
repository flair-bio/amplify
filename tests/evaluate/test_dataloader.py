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
from modules.evaluate.src.dataset.collator import EvaluationCollator, EvaluationCollatorConfig
from modules.evaluate.src.dataset.dataloader import DataLoaderConfig, LengthGroupedSampler, TokenBudgetBatchSampler, build_dataloaders, ensure_validation_split, load_evaluation_dataset, select_dataset_splits, tokenize_dataset, token_lengths
from modules.evaluate.src.contracts.config import DatasetSplitConfig
from modules.evaluate.src.utils.seed import seed_everything
from modules.pretrain.src.model.tokenizer import ProteinTokenizer, TokenizerConfig

from .conftest import LONG_SEQUENCE, SHORT_SEQUENCE, VOCAB, build_raw_dataset_dict


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
# DataLoaderConfig
# ---------------------------------------------------------------------------


class TestDataLoaderConfig:
    def test_default_worker_count(self):
        assert DataLoaderConfig().num_workers == 8

    @pytest.mark.parametrize("num_workers", [0, 1, 16])
    def test_explicit_worker_count(self, num_workers):
        assert DataLoaderConfig(num_workers=num_workers).num_workers == num_workers

    @pytest.mark.parametrize("num_workers", [-1, None])
    def test_invalid_worker_count(self, num_workers):
        with pytest.raises(ValueError):
            DataLoaderConfig(num_workers=num_workers)


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

    def test_preserves_only_configured_final_refit_splits(self):
        raw = build_raw_dataset_dict("sequence_classification")
        raw["tuning_excluded"] = raw["train"].select([0])

        result = select_dataset_splits(
            raw,
            DatasetSplitConfig(refit_additional_splits=["tuning_excluded"]),
        )

        assert set(result) == {
            "train",
            "validation",
            "test",
            "tuning_excluded",
        }
        assert result["tuning_excluded"] is raw["tuning_excluded"]

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

    def test_rejects_final_refit_split_that_repeats_a_role(self):
        with pytest.raises(ValueError, match="must not repeat"):
            DatasetSplitConfig(refit_additional_splits=["test"])

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


def test_contact_input_check_preserves_generic_tokenization_diagnostic(
    protein_tokenizer,
):
    raw = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sequence": [SHORT_SEQUENCE],
                    "targets": [0],
                    "positive_pairs": [[]],
                }
            )
        }
    )

    with pytest.raises(
        ValueError, match="CASP scoring rows require tokenize_contact_chain"
    ):
        tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=32,
        )


def test_missing_source_label_reports_split_and_column(protein_tokenizer):
    raw = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "sequence": [SHORT_SEQUENCE],
                    "targets": [[0] * len(SHORT_SEQUENCE)],
                }
            ),
            "validation": Dataset.from_dict({"sequence": [SHORT_SEQUENCE]}),
        }
    )

    with pytest.raises(
        ValueError,
        match="Dataset split 'validation' is missing required label column 'targets'",
    ):
        tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=32,
            num_proc=1,
        )


@pytest.mark.parametrize("num_proc", [1, 2])
@pytest.mark.parametrize("tokenizer_fixture", ["protein_tokenizer", "esm2_tokenizer"])
def test_pseudo_perplexity_bounds_expansion_batches(
    request, tokenizer_fixture, num_proc, monkeypatch
):
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    sequences = [SHORT_SEQUENCE, "A" * 600, "MKT"]
    source_ids = ["a", "b", "a"]
    raw = DatasetDict(
        {"test": Dataset.from_dict({"sequence": sequences, "id": source_ids})}
    )
    original_map = Dataset.map

    def bounded_map(self, function, **kwargs):
        assert kwargs["batch_size"] == 1

        def checked_batch(examples, indices, **fn_kwargs):
            assert len(examples["sequence"]) <= 1
            return function(examples, indices, **fn_kwargs)

        return original_map(self, checked_batch, **kwargs)

    monkeypatch.setattr(Dataset, "map", bounded_map)
    result = tokenize_dataset(
        raw,
        tokenizer=tokenizer,
        task_type="pseudo_perplexity",
        sequence_column="sequence",
        label_column="targets",
        max_length=512,
        num_proc=num_proc,
        random_truncate=False,
    )["test"]

    assert len(result) == len(SHORT_SEQUENCE) + 510 + 3
    row_index = 0
    expanded_ids = []
    for source_index, (sequence, example_id) in enumerate(zip(sequences, source_ids)):
        residue_ids = tokenizer(sequence, add_special_tokens=False)["input_ids"][:510]
        leading_id = tokenizer.bos_token_id
        if leading_id is None:
            leading_id = tokenizer.cls_token_id
        input_ids = [leading_id] + residue_ids + [tokenizer.eos_token_id]
        for position in range(1, len(residue_ids) + 1):
            row = result[row_index]
            expected_ids = list(input_ids)
            expected_ids[position] = tokenizer.mask_token_id
            expected_labels = [-100] * len(input_ids)
            expected_labels[position] = input_ids[position]
            assert row["input_ids"] == expected_ids
            assert row["targets"] == expected_labels
            assert row["attention_mask"] == [1] * len(input_ids)
            assert row["id"] == f"{example_id}:{source_index}:{position - 1}"
            expanded_ids.append(row["id"])
            row_index += 1
    assert len(expanded_ids) == len(set(expanded_ids))


def test_contact_tokenization_preserves_valid_positions_and_length(protein_tokenizer):
    raw = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sequence": ["A" * 15],
                    "targets": [[[2, 9]]],
                    "valid_positions": [[2, 9, 12]],
                }
            )
        }
    )

    row = tokenize_dataset(
        raw,
        tokenizer=protein_tokenizer,
        task_type="contact_prediction",
        sequence_column="sequence",
        label_column="targets",
        max_length=20,
        num_proc=1,
        random_truncate=False,
    )["test"][0]

    assert row["targets"] == {
        "pairs": [[3, 10]],
        "valid_positions": [3, 10, 13],
        "res_start": 1,
        "res_end": 16,
        "length": 15,
    }
    collator = EvaluationCollator(
        tokenizer=protein_tokenizer,
        task_type="contact_prediction",
        config=EvaluationCollatorConfig(label_column="targets"),
    )
    batch = collator([row])
    assert batch["residue_lengths"].tolist() == [15]
    assert batch["labels"][0, 3, 10] == 1
    assert batch["labels"][0, 3, 9] == -100
    assert batch["labels"][0, 3, 13] == 0


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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
        )
        example = tokenized["train"][0]
        assert len(example["input_ids"]) == len(SHORT_SEQUENCE) + 2


class TestTokenizeDatasetTokenClassification:
    @pytest.mark.parametrize("random_truncate", [False, True])
    @pytest.mark.parametrize("max_length", [7, 64])
    @pytest.mark.parametrize("num_proc", [1, 2])
    def test_preserves_ambiguous_residue_coordinates(
        self, tmp_path, random_truncate, max_length, num_proc
    ):
        tokenizer = ProteinTokenizer(
            TokenizerConfig(
                vocab=VOCAB + ["X", "B", "Z"],
                pad_token="<pad>",
                unk_token="<unk>",
                mask_token="<mask>",
                bos_token="<bos>",
                eos_token="<eos>",
                ambiguous_tokens=["X", "B", "Z"],
                remove_ambiguous=True,
            )
        )
        sequences = ["XACBDZ", "ZBXCAX", "AXBCDX", "XAJCDZ"]
        raw = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "sequence": sequences,
                        "targets": [list(range(len(seq))) for seq in sequences],
                    }
                )
            }
        )
        tokenized = tokenize_dataset(
            raw,
            tokenizer=tokenizer,
            task_type="token_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=max_length,
            random_truncate=random_truncate,
            num_proc=num_proc,
        )
        assert tokenizer.remove_ambiguous is False
        for sequence, example in zip(sequences, tokenized["train"]):
            assert example["targets"][0] == example["targets"][-1] == -100
            positions = example["targets"][1:-1]
            assert len(positions) == min(len(sequence), max_length - 2)
            assert example["input_ids"][1:-1] == [
                tokenizer.convert_tokens_to_ids(sequence[position])
                for position in positions
            ]

        tokenizer.save_pretrained(tmp_path / "tokenizer")
        restored = ProteinTokenizer.from_pretrained(tmp_path / "tokenizer")
        assert restored.remove_ambiguous is False
        assert len(restored("XACBDZ", add_special_tokens=False)["input_ids"]) == 6

    def test_sequence_tasks_keep_checkpoint_filtering(self, protein_tokenizer):
        protein_tokenizer.remove_ambiguous = True
        protein_tokenizer.ambiguous_token_ids = [
            protein_tokenizer.convert_tokens_to_ids("A")
        ]
        raw = DatasetDict(
            {"train": Dataset.from_dict({"sequence": ["MAC"], "targets": [0]})}
        )
        tokenized = tokenize_dataset(
            raw,
            tokenizer=protein_tokenizer,
            task_type="sequence_classification",
            sequence_column="sequence",
            label_column="targets",
            max_length=64,
            num_proc=1,
        )
        assert protein_tokenizer.remove_ambiguous is True
        assert tokenized["train"][0]["input_ids"][1:-1] == [
            protein_tokenizer.convert_tokens_to_ids(residue) for residue in "MC"
        ]

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
                num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
            num_proc=1,
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
    def test_eval_order_is_stable_across_passes(self):
        lengths = [5] * 10
        sampler = LengthGroupedSampler(
            lengths,
            batch_size=2,
            generator=torch.Generator().manual_seed(42),
            shuffle=False,
        )
        first = list(sampler)
        assert first == list(sampler)
        assert first == sorted(
            range(len(lengths)), key=lambda index: lengths[index], reverse=True
        )

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
            num_proc=1,
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


@pytest.mark.parametrize("format_type", [None, "numpy", "torch"])
@pytest.mark.parametrize("indices", [None, [3, 0, 2, 2], []])
def test_token_lengths_preserve_order_and_format(format_type, indices):
    dataset = Dataset.from_dict({"input_ids": [[1, 2], [], [3, 4, 5], [6]]})
    if indices is not None:
        dataset = dataset.select(indices)
    dataset = dataset.with_format(format_type)
    original_format = dataset.format

    assert token_lengths(dataset) == [len(ids) for ids in dataset["input_ids"]]
    assert dataset.format == original_format


@pytest.mark.parametrize("budget", [None, 8])
def test_column_lengths_preserve_batch_assignments(budget):
    dataset = Dataset.from_dict(
        {"input_ids": [[0] * length for length in [4, 2, 5, 3, 4, 1]]}
    ).select([4, 0, 3, 2, 1, 5])
    lengths = [len(ids) for ids in dataset["input_ids"]]
    train, validation, test = build_dataloaders(
        DatasetDict({role: dataset for role in ("train", "validation", "test")}),
        collator=lambda batch: batch,
        config=DataLoaderConfig(
            batch_size=2,
            num_workers=0,
            length_grouped_sampling=True,
            max_tokens_per_batch=budget,
            seed=17,
        ),
    )
    for loader, shuffle in [(train, True), (validation, False), (test, False)]:
        generator = torch.Generator().manual_seed(17)
        if budget is None:
            sampler = LengthGroupedSampler(lengths, 2, generator, shuffle)
            order = list(sampler)
            expected = [order[index : index + 2] for index in range(0, len(order), 2)]
        else:
            expected = list(
                TokenBudgetBatchSampler(lengths, budget, shuffle, generator)
            )
        assert list(loader.batch_sampler) == expected
