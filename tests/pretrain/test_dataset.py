"""Unit tests for dataset curriculum, cluster splitting, and per-epoch sampling.

These cover the correctness-critical data logic that the end-to-end smoke test
only exercises incidentally: the deterministic train/val holdout (a leak here
silently invalidates every reported metric) and the score curriculum.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from modules.pretrain.src.dataset.dataset import (
    ClusterIndex,
    CurriculumConfig,
    _cluster_hash,
    _cluster_split_filter,
    build_source_epoch_plan,
    compute_or_load_valid_counts,
    curriculum_score_threshold,
    load_or_build_cluster_index,
    pick_one_valid_row_per_cluster,
    stage_dataset_locally,
)

_CLUSTER_COL = "cluster_rep_at_30"


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------


class TestCurriculumScoreThreshold:
    @pytest.mark.parametrize("kind", ["linear", "cosine", "exponential"])
    def test_starts_at_start_score_and_plateaus_at_end_score(self, kind):
        curriculum = CurriculumConfig(
            type=kind, start_score=0.1, end_score=0.8, ramp_epochs=4
        )

        assert curriculum_score_threshold(curriculum, 0) == pytest.approx(0.1)
        assert curriculum_score_threshold(curriculum, 4) == pytest.approx(0.8)
        # Past the ramp the threshold holds rather than overshooting.
        assert curriculum_score_threshold(curriculum, 99) == pytest.approx(0.8)

    @pytest.mark.parametrize("kind", ["linear", "cosine", "exponential"])
    def test_is_monotonic_across_the_ramp(self, kind):
        curriculum = CurriculumConfig(
            type=kind, start_score=0.1, end_score=0.8, ramp_epochs=6
        )
        values = [curriculum_score_threshold(curriculum, e) for e in range(8)]

        assert all(b >= a for a, b in zip(values, values[1:]))

    def test_linear_midpoint_is_the_arithmetic_mean(self):
        curriculum = CurriculumConfig(
            type="linear", start_score=0.0, end_score=1.0, ramp_epochs=2
        )
        assert curriculum_score_threshold(curriculum, 1) == pytest.approx(0.5)

    def test_cosine_midpoint_matches_the_closed_form(self):
        curriculum = CurriculumConfig(
            type="cosine", start_score=0.0, end_score=1.0, ramp_epochs=4
        )
        expected = (1 - math.cos(math.pi * 0.25)) / 2
        assert curriculum_score_threshold(curriculum, 1) == pytest.approx(expected)

    def test_exponential_midpoint_is_the_geometric_mean(self):
        curriculum = CurriculumConfig(
            type="exponential", start_score=0.25, end_score=1.0, ramp_epochs=2
        )
        assert curriculum_score_threshold(curriculum, 1) == pytest.approx(0.5)

    def test_exponential_rejects_a_zero_start(self):
        # start * (end/start)**p is undefined at zero.
        with pytest.raises(ValueError):
            CurriculumConfig(
                type="exponential", start_score=0.0, end_score=1.0, ramp_epochs=2
            )


# ---------------------------------------------------------------------------
# Deterministic train/val holdout
# ---------------------------------------------------------------------------


class TestClusterSplit:
    def test_hash_is_stable_across_calls(self):
        assert _cluster_hash("cluster_a") == _cluster_hash("cluster_a")

    def test_hash_is_not_python_salted_hash(self):
        """Must survive process restarts and differ across ranks' interpreters.

        `hash()` is randomized per process by PYTHONHASHSEED, which would make
        the holdout differ between runs and between DDP ranks.
        """
        import hashlib

        expected = int(hashlib.md5(b"cluster_a").hexdigest(), 16)
        assert _cluster_hash("cluster_a") == expected

    def test_int_and_str_cluster_ids_agree(self):
        assert _cluster_hash(42) == _cluster_hash("42")

    def test_train_and_val_are_exact_complements(self):
        batch = {_CLUSTER_COL: [f"c{i}" for i in range(200)]}

        val = _cluster_split_filter(batch, _CLUSTER_COL, modulus=5, keep_val=True)
        train = _cluster_split_filter(batch, _CLUSTER_COL, modulus=5, keep_val=False)

        # No row is dropped and none is claimed by both splits.
        assert [v ^ t for v, t in zip(val, train)] == [True] * 200

    def test_no_cluster_appears_in_both_splits(self):
        clusters = [f"c{i}" for i in range(300)]
        batch = {_CLUSTER_COL: clusters}

        val = _cluster_split_filter(batch, _CLUSTER_COL, modulus=5, keep_val=True)
        train = _cluster_split_filter(batch, _CLUSTER_COL, modulus=5, keep_val=False)

        val_ids = {c for c, keep in zip(clusters, val) if keep}
        train_ids = {c for c, keep in zip(clusters, train) if keep}
        assert val_ids.isdisjoint(train_ids)

    def test_repeated_cluster_id_always_lands_on_the_same_side(self):
        """Homologous rows share a cluster; splitting them would leak."""
        batch = {_CLUSTER_COL: ["c7"] * 10 + ["c8"] * 10}

        val = _cluster_split_filter(batch, _CLUSTER_COL, modulus=3, keep_val=True)

        assert len(set(val[:10])) == 1
        assert len(set(val[10:])) == 1

    def test_null_cluster_ids_are_kept_for_training(self):
        batch = {_CLUSTER_COL: [None, None]}

        assert _cluster_split_filter(batch, _CLUSTER_COL, 5, keep_val=True) == [
            False,
            False,
        ]
        assert _cluster_split_filter(batch, _CLUSTER_COL, 5, keep_val=False) == [
            True,
            True,
        ]

    def test_modulus_sets_the_holdout_rate(self):
        batch = {_CLUSTER_COL: [f"c{i}" for i in range(4000)]}

        val = _cluster_split_filter(batch, _CLUSTER_COL, modulus=10, keep_val=True)

        # MD5 spreads uniformly; allow slack for a finite sample.
        assert 0.075 < sum(val) / len(val) < 0.125


# ---------------------------------------------------------------------------
# Per-epoch cluster sampling
# ---------------------------------------------------------------------------


def _index(clusters: list[int], scores: list[float]) -> ClusterIndex:
    """Build an in-memory ClusterIndex sorted by (cluster, score desc)."""
    order = np.lexsort((-np.asarray(scores), np.asarray(clusters)))
    sorted_clusters = np.asarray(clusters)[order]
    boundaries = np.concatenate(
        [[0], np.flatnonzero(np.diff(sorted_clusters)) + 1, [len(order)]]
    ).astype(np.int64)
    return ClusterIndex(
        order=order.astype(np.int64),
        boundaries=boundaries,
        scores=np.asarray(scores)[order].astype(np.float32),
        lengths=np.full(len(order), 32, dtype=np.int64),
        cache_dir=None,
    )


class TestValidCounts:
    def test_counts_rows_at_or_above_the_threshold_per_cluster(self):
        index = _index([0, 0, 0, 1, 1], [0.9, 0.5, 0.1, 0.8, 0.2])

        counts = compute_or_load_valid_counts(index, threshold=0.5)

        assert counts.tolist() == [2, 1]

    def test_threshold_is_inclusive(self):
        index = _index([0, 0], [0.5, 0.4])
        assert compute_or_load_valid_counts(index, threshold=0.5).tolist() == [1]

    def test_threshold_above_every_score_yields_zero(self):
        index = _index([0, 0, 1], [0.9, 0.5, 0.8])
        assert compute_or_load_valid_counts(index, threshold=1.0).tolist() == [0, 0]

    def test_empty_index_returns_empty(self):
        index = ClusterIndex(
            order=np.array([], dtype=np.int64),
            boundaries=np.array([0], dtype=np.int64),
            scores=np.array([], dtype=np.float32),
            lengths=np.array([], dtype=np.int64),
            cache_dir=None,
        )
        assert compute_or_load_valid_counts(index, threshold=0.5).size == 0

    def test_counts_are_cached_to_disk_and_reused(self, tmp_path):
        index = _index([0, 0, 1], [0.9, 0.1, 0.8])
        index.cache_dir = tmp_path

        first = compute_or_load_valid_counts(index, threshold=0.5)
        cached = list(tmp_path.glob("valid_count-*.npy"))
        second = compute_or_load_valid_counts(index, threshold=0.5)

        assert len(cached) == 1
        assert first.tolist() == second.tolist()


class TestPickOneValidRowPerCluster:
    def test_picks_exactly_one_row_from_each_eligible_cluster(self):
        index = _index([0, 0, 1, 1, 2], [0.9, 0.8, 0.7, 0.6, 0.5])
        counts = compute_or_load_valid_counts(index, threshold=0.0)

        rows = pick_one_valid_row_per_cluster(
            index, counts, np.random.default_rng(0)
        )

        assert len(rows) == 3
        assert len(set(rows.tolist())) == 3

    def test_clusters_without_valid_rows_are_skipped(self):
        index = _index([0, 0, 1], [0.9, 0.8, 0.1])
        counts = compute_or_load_valid_counts(index, threshold=0.5)

        rows = pick_one_valid_row_per_cluster(
            index, counts, np.random.default_rng(0)
        )

        # Only cluster 0 has a qualifying row.
        assert len(rows) == 1
        assert rows[0] in {0, 1}

    def test_only_rows_meeting_the_threshold_are_selected(self):
        scores = [0.9, 0.8, 0.1, 0.05]
        index = _index([0, 0, 0, 0], scores)
        counts = compute_or_load_valid_counts(index, threshold=0.5)

        picked = {
            int(pick_one_valid_row_per_cluster(
                index, counts, np.random.default_rng(seed)
            )[0])
            for seed in range(25)
        }

        assert picked <= {0, 1}

    def test_selection_is_reproducible_for_a_given_seed(self):
        index = _index([0, 0, 1, 1], [0.9, 0.8, 0.7, 0.6])
        counts = compute_or_load_valid_counts(index, threshold=0.0)

        a = pick_one_valid_row_per_cluster(index, counts, np.random.default_rng(7))
        b = pick_one_valid_row_per_cluster(index, counts, np.random.default_rng(7))

        assert a.tolist() == b.tolist()

    def test_resampling_varies_across_epochs(self):
        index = _index([0] * 8, [1.0] * 8)
        counts = compute_or_load_valid_counts(index, threshold=0.0)

        picked = {
            int(pick_one_valid_row_per_cluster(
                index, counts, np.random.default_rng(seed)
            )[0])
            for seed in range(30)
        }

        assert len(picked) > 1


class TestBuildSourceEpochPlan:
    """Per-source plan builder: one representative row per eligible cluster."""

    def test_clusters_without_valid_rows_are_excluded(self):
        index = ClusterIndex(
            order=np.array([0, 1, 2]),
            boundaries=np.array([0, 1, 2, 3]),
            scores=np.array([1.0, 1.0, 0.0]),
            lengths=np.array([10, 20, 30]),
        )
        valid_counts = compute_or_load_valid_counts(index, 0.5)

        plan = build_source_epoch_plan(
            index, valid_counts, fraction=1.0, rng=np.random.default_rng(0)
        )

        assert sorted(plan.row_ids.tolist()) == [0, 1]

    def test_sequence_lengths_stay_aligned_with_row_ids(self):
        """Misalignment silently corrupts token-budget batching."""
        index = ClusterIndex(
            order=np.array([2, 0, 1]),
            boundaries=np.array([0, 1, 2, 3]),
            scores=np.array([1.0, 1.0, 1.0]),
            lengths=np.array([30, 10, 20]),
        )
        valid_counts = compute_or_load_valid_counts(index, 0.5)

        plan = build_source_epoch_plan(
            index, valid_counts, fraction=1.0, rng=np.random.default_rng(0)
        )

        expected = {2: 30, 0: 10, 1: 20}
        for row_id, length in zip(plan.row_ids, plan.sequence_lengths):
            assert length == expected[int(row_id)]

    def test_no_valid_rows_returns_empty_plan(self):
        index = ClusterIndex(
            order=np.array([0, 1]),
            boundaries=np.array([0, 1, 2]),
            scores=np.array([0.0, 0.0]),
            lengths=np.array([10, 10]),
        )
        valid_counts = compute_or_load_valid_counts(index, 0.5)

        plan = build_source_epoch_plan(
            index, valid_counts, fraction=1.0, rng=np.random.default_rng(0)
        )

        assert plan.row_ids.size == 0
        assert plan.sequence_lengths.size == 0

    def test_fraction_subsamples_eligible_clusters_without_duplicates(self):
        n = 20
        index = ClusterIndex(
            order=np.arange(n, dtype=np.int64),
            boundaries=np.arange(n + 1, dtype=np.int64),  # every row its own cluster
            scores=np.ones(n),
            lengths=np.arange(n, dtype=np.int64) * 10,
        )
        valid_counts = compute_or_load_valid_counts(index, 0.5)

        plan = build_source_epoch_plan(
            index, valid_counts, fraction=0.3, rng=np.random.default_rng(0)
        )

        assert len(plan.row_ids) == round(n * 0.3)
        assert len(set(plan.row_ids.tolist())) == len(plan.row_ids)

    def test_matches_naive_reference_on_random_inputs(self):
        """Every returned row must be a legal (score >= threshold) pick for its cluster."""
        rng = np.random.default_rng(0)
        for trial in range(100):
            n = int(rng.integers(1, 20))
            num_cuts = int(rng.integers(0, n))
            cuts = (
                np.sort(rng.choice(np.arange(1, n), size=num_cuts, replace=False))
                if n > 1
                else np.array([], dtype=np.int64)
            )
            boundaries = np.concatenate(([0], cuts, [n])).astype(np.int64)
            scores = rng.choice([0.0, 1.0], size=n, p=[0.6, 0.4])
            for s, e in zip(boundaries[:-1], boundaries[1:]):
                scores[s:e] = np.sort(scores[s:e])[::-1]
            index = ClusterIndex(
                order=rng.permutation(n).astype(np.int64),
                boundaries=boundaries,
                scores=scores,
                lengths=np.arange(n, dtype=np.int64),
            )
            valid_counts = compute_or_load_valid_counts(index, 0.5)

            plan = build_source_epoch_plan(
                index, valid_counts, fraction=1.0, rng=np.random.default_rng(trial)
            )

            legal_choices = [
                {index.order[i] for i in range(s, e) if index.scores[i] >= 0.5}
                for s, e in zip(boundaries[:-1], boundaries[1:])
            ]
            legal_choices = [c for c in legal_choices if c]
            assert len(plan.row_ids) == len(legal_choices)
            assert all(
                row in choices for row, choices in zip(plan.row_ids, legal_choices)
            )


# ---------------------------------------------------------------------------
# Cluster index cache
# ---------------------------------------------------------------------------


def _hf_dataset(num_rows: int = 200):
    """Small in-memory dataset shaped like a prepared source."""
    from datasets import Dataset as HFDataset

    return HFDataset.from_dict(
        {
            "sequence": ["MKT" * 5] * num_rows,
            _CLUSTER_COL: [f"C{i % 20:06d}" for i in range(num_rows)],
            "red_score": [(i % 10) / 10 for i in range(num_rows)],
            "sequence_length": [15] * num_rows,
        }
    )


class TestClusterIndexCache:
    """``order`` indexes positionally, so a wrong hit corrupts data silently."""

    def test_identical_dataset_hits_the_cache(self, tmp_path):
        built = load_or_build_cluster_index(
            _hf_dataset(), _CLUSTER_COL, "red_score", tmp_path
        )
        reused = load_or_build_cluster_index(
            _hf_dataset(), _CLUSTER_COL, "red_score", tmp_path
        )

        assert reused.cache_dir == built.cache_dir
        assert np.array_equal(np.asarray(reused.order), np.asarray(built.order))

    def test_changed_fingerprint_rebuilds(self, tmp_path):
        """The fingerprint is what makes the cache self-invalidating."""
        built = load_or_build_cluster_index(
            _hf_dataset(), _CLUSTER_COL, "red_score", tmp_path
        )
        changed = _hf_dataset()
        changed._fingerprint = "a-different-fingerprint"

        rebuilt = load_or_build_cluster_index(
            changed, _CLUSTER_COL, "red_score", tmp_path
        )

        assert rebuilt.cache_dir != built.cache_dir

    def test_row_count_mismatch_rebuilds_instead_of_returning_stale_index(
        self, tmp_path
    ):
        """Guards a hand-copied or truncated cache dir: without this, `scores`
        and `lengths` would be paired with the wrong rows."""
        dataset = _hf_dataset(num_rows=200)
        index = load_or_build_cluster_index(
            dataset, _CLUSTER_COL, "red_score", tmp_path
        )

        # Same cache dir, but the dataset underneath it now has more rows.
        larger = _hf_dataset(num_rows=400)
        larger._fingerprint = dataset._fingerprint
        reloaded = load_or_build_cluster_index(
            larger, _CLUSTER_COL, "red_score", tmp_path
        )

        assert reloaded.cache_dir == index.cache_dir
        assert len(np.asarray(reloaded.order)) == 400


# ---------------------------------------------------------------------------
# Local staging
# ---------------------------------------------------------------------------


class TestStageDatasetLocally:
    """Row ids and the cluster index point into the original row order, so a
    staged copy must preserve it exactly."""

    def test_copy_preserves_rows_and_order(self, tmp_path):
        dataset = _hf_dataset().select_columns(["sequence", _CLUSTER_COL])

        staged = stage_dataset_locally(dataset, tmp_path / "src-train")

        assert staged[:] == dataset[:]
        assert all(
            Path(f["filename"]).is_relative_to(tmp_path) for f in staged.cache_files
        )

    def test_existing_copy_is_reused(self, tmp_path):
        dataset = _hf_dataset()
        stage_dataset_locally(dataset, tmp_path / "src-train")
        (target,) = [p for p in tmp_path.iterdir() if p.is_dir()]
        mtime = target.stat().st_mtime_ns

        reused = stage_dataset_locally(dataset, tmp_path / "src-train")

        assert target.stat().st_mtime_ns == mtime
        assert len(reused) == len(dataset)

    def test_row_count_mismatch_restages(self, tmp_path):
        small = _hf_dataset(num_rows=100)
        stage_dataset_locally(small, tmp_path / "src-train")
        larger = _hf_dataset(num_rows=300)
        larger._fingerprint = small._fingerprint

        restaged = stage_dataset_locally(larger, tmp_path / "src-train")

        assert len(restaged) == 300
        assert not list(tmp_path.glob("*.tmp-*"))
