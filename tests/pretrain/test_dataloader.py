"""Unit tests for token-budget batching, rank truncation, and worker seeding.

These cover the sampler logic that determines what actually reaches the GPU:
a packing bug silently changes the effective batch size (and therefore the
learning-rate schedule), and a rank-truncation bug deadlocks multi-GPU runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.pretrain.src.dataset.dataloader import (
    DataLoaderConfig,
    TokenBatchSampler,
    _drop_for_even_ranks,
    build_epoch_dataloader,
    build_worker_seed_generator,
)
from modules.pretrain.src.dataset.dataset import EpochPlan

# Every sequence costs its length plus BOS and EOS.
_SPECIAL_TOKENS = 2


def _batch_cost(batch: list[int], lengths: dict[int, int]) -> int:
    return sum(lengths[r] + _SPECIAL_TOKENS for r in batch)


class TestTokenBatchSampler:
    def test_every_row_appears_exactly_once_and_in_order(self):
        row_ids = [10, 11, 12, 13, 14, 15, 16]
        lengths = [8, 3, 12, 5, 9, 2, 7]

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=20))

        assert [r for b in batches for r in b] == row_ids

    def test_batches_stay_within_the_token_budget(self):
        row_ids = list(range(40))
        rng = np.random.default_rng(0)
        lengths = rng.integers(1, 30, size=40).tolist()
        by_row = dict(zip(row_ids, lengths))

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=64))

        assert all(_batch_cost(b, by_row) <= 64 for b in batches)

    def test_packing_is_greedy_not_one_row_per_batch(self):
        """Rows must be packed until the budget is hit, not emitted singly."""
        row_ids = list(range(6))
        lengths = [4] * 6  # cost 6 each, so 5 fit in a 32-token budget

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=32))

        assert batches == [[0, 1, 2, 3, 4], [5]]

    def test_budget_boundary_is_inclusive(self):
        row_ids = [0, 1, 2]
        lengths = [8, 8, 8]  # cost 10 each

        # Exactly 20 tokens fit two rows; one token less fits only one.
        assert list(TokenBatchSampler(row_ids, lengths, max_tokens=20))[0] == [0, 1]
        assert list(TokenBatchSampler(row_ids, lengths, max_tokens=19))[0] == [0]

    def test_sequence_larger_than_the_budget_gets_its_own_batch(self):
        """Without the ``max(j, i + 1)`` clamp this would stall or drop rows."""
        row_ids = [0, 1, 2]
        lengths = [4, 500, 4]

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=32))

        assert [1] in batches
        assert [r for b in batches for r in b] == row_ids

    def test_accounts_for_bos_and_eos(self):
        """A budget that fits raw lengths but not the special tokens must split."""
        row_ids = [0, 1]
        lengths = [5, 5]  # raw 10, but real cost 14

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=12))

        assert batches == [[0], [1]]

    def test_row_ids_are_global_not_positional(self):
        """Batches must yield dataset row ids, not positions into the plan."""
        row_ids = [100, 200, 300]
        lengths = [2, 2, 2]

        batches = list(TokenBatchSampler(row_ids, lengths, max_tokens=1000))

        assert batches == [[100, 200, 300]]

    def test_empty_plan_yields_no_batches(self):
        sampler = TokenBatchSampler([], [], max_tokens=64)

        assert len(sampler) == 0
        assert list(sampler) == []

    def test_len_matches_the_number_of_batches_yielded(self):
        rng = np.random.default_rng(1)
        lengths = rng.integers(1, 40, size=97).tolist()

        sampler = TokenBatchSampler(list(range(97)), lengths, max_tokens=128)

        assert len(sampler) == len(list(sampler))

    def test_iteration_is_repeatable(self):
        sampler = TokenBatchSampler(list(range(20)), [6] * 20, max_tokens=40)

        assert list(sampler) == list(sampler)

    def test_costs_are_capped_at_max_sequence_length(self):
        # Raw lengths far exceed the cap; each truncates to 10 tokens, so all
        # four fit in a 40-token budget instead of landing in separate batches.
        row_ids = [0, 1, 2, 3]
        lengths = [1000, 1000, 1000, 1000]

        batches = list(
            TokenBatchSampler(
                row_ids, lengths, max_tokens=40, max_sequence_length=10
            )
        )

        assert batches == [[0, 1, 2, 3]]

    def test_short_sequences_are_unaffected_by_the_cap(self):
        row_ids = list(range(6))
        lengths = [8, 3, 12, 5, 9, 2]

        uncapped = list(TokenBatchSampler(row_ids, lengths, max_tokens=20))
        capped = list(
            TokenBatchSampler(
                row_ids, lengths, max_tokens=20, max_sequence_length=512
            )
        )

        assert capped == uncapped

    def test_cap_defaults_to_no_truncation(self):
        row_ids = [0, 1]
        lengths = [100, 100]

        assert list(TokenBatchSampler(row_ids, lengths, max_tokens=50)) == [[0], [1]]


class TestBuildEpochDataloaderPassesCollatorMaxLength:
    def test_batches_use_the_collators_truncation_length(self):
        class _StubCollator:
            max_length = 10

            def __call__(self, samples):
                return samples

        plan = EpochPlan(
            row_ids=np.arange(4),
            sequence_lengths=np.array([1000, 1000, 1000, 1000]),
        )

        loader = build_epoch_dataloader(
            concat_dataset=[0, 1, 2, 3],
            epoch_plan=plan,
            collator=_StubCollator(),
            dataloader_config=DataLoaderConfig(
                max_tokens=40, dataloader_num_workers=0, drop_last=False
            ),
        )

        assert list(loader.batch_sampler) == [[0, 1, 2, 3]]


class TestBuildEpochDataloaderInOrder:
    @pytest.mark.parametrize("in_order", [True, False])
    def test_in_order_is_forwarded(self, in_order):
        plan = EpochPlan(row_ids=np.arange(4), sequence_lengths=np.ones(4))

        loader = build_epoch_dataloader(
            concat_dataset=[0, 1, 2, 3],
            epoch_plan=plan,
            collator=lambda samples: samples,
            dataloader_config=DataLoaderConfig(
                batch_size=2, dataloader_num_workers=0, in_order=in_order
            ),
        )

        assert loader.in_order is in_order


class TestDropForEvenRanks:
    def test_single_process_is_a_passthrough(self):
        batches = [[0], [1], [2]]

        assert _drop_for_even_ranks(batches, num_processes=1) is batches

    def test_already_divisible_input_is_returned_unchanged(self):
        batches = [[0], [1], [2], [3]]

        assert _drop_for_even_ranks(batches, num_processes=4) is batches

    def test_truncates_to_a_multiple_of_the_process_count(self):
        batches = [[i] for i in range(11)]

        result = _drop_for_even_ranks(batches, num_processes=4)

        assert len(result) == 8
        assert len(result) % 4 == 0

    def test_keeps_the_leading_batches(self):
        batches = [[i] for i in range(7)]

        result = _drop_for_even_ranks(batches, num_processes=2)

        assert list(result) == batches[:6]

    def test_truncates_a_lazy_sampler_without_materializing_it(self):
        sampler = TokenBatchSampler(list(range(20)), [6] * 20, max_tokens=40)
        assert len(sampler) % 3 != 0, "fixture must need truncation"

        result = _drop_for_even_ranks(sampler, num_processes=3)

        assert not isinstance(result, list)
        assert len(result) % 3 == 0
        assert list(result) == list(sampler)[: len(result)]


class TestWorkerSeedGenerator:
    def test_is_deterministic_for_identical_keys(self):
        a = build_worker_seed_generator(seed=42, epoch=0)
        b = build_worker_seed_generator(seed=42, epoch=0)

        assert a.initial_seed() == b.initial_seed()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"seed": 43, "epoch": 0},
            {"seed": 42, "epoch": 1},
            {"seed": 42, "epoch": 0, "split": "val"},
            {"seed": 42, "epoch": 0, "process_index": 1},
        ],
        ids=["seed", "epoch", "split", "rank"],
    )
    def test_each_key_component_changes_the_seed(self, kwargs):
        """Every component gets its own slot; none may alias another."""
        base = build_worker_seed_generator(seed=42, epoch=0)

        assert build_worker_seed_generator(**kwargs).initial_seed() != (
            base.initial_seed()
        )

    def test_is_independent_of_ambient_global_rng_state(self):
        import torch

        torch.manual_seed(1234)
        first = build_worker_seed_generator(seed=42, epoch=0).initial_seed()
        torch.manual_seed(99)
        second = build_worker_seed_generator(seed=42, epoch=0).initial_seed()

        assert first == second

    def test_unknown_split_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown split"):
            build_worker_seed_generator(seed=42, epoch=0, split="test")
