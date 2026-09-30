"""Tests for ProteinGym zero-shot masked-marginal scoring and aggregation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from modules.evaluate.scripts.dataset_sourcing.source_proteingym import (
    _add_seeded_splits,
    _load_assay,
    _TRACKS,
    write_proteingym_card,
)
from modules.evaluate.src.proteingym.aggregate import (
    aggregate_proteingym_metrics,
    aggregate_proteingym_scores,
    bootstrap_standard_error_by_category,
    compute_all_assay_metrics,
    compute_assay_auc,
    compute_assay_mcc,
    compute_assay_metric,
    compute_assay_ndcg,
    compute_assay_spearman,
    compute_assay_top_recall,
)
from modules.evaluate.src.proteingym.baselines import (
    score_assay_blosum62,
    score_assay_random,
)
from modules.evaluate.src.proteingym.masked_marginal import (
    _optimal_window,
    parse_mutant,
    score_assay_indel,
    score_assay_masked_marginal,
)
from modules.evaluate.src.proteingym.supervised import (
    OFFICIAL_CV_SCHEMES,
    pool_embeddings,
    resolve_folds,
    run_cv,
)
from modules.evaluate.src.proteingym import runner as proteingym_runner
from modules.evaluate.src.proteingym.runner import run_assay_loop

_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class _FakeTokenizer:
    """Deterministic single-char-per-residue tokenizer with a leading CLS."""

    def __init__(self) -> None:
        self._token_to_id = {aa: i + 3 for i, aa in enumerate(_ALPHABET)}
        self.cls_token_id = 1
        self.mask_token_id = 2
        self.vocab_size = len(self._token_to_id) + 3

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id[token]

    def __call__(
        self,
        sequence,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        sequences = [sequence] if isinstance(sequence, str) else sequence
        # Reserve one slot for CLS when truncating so callers can't overrun max_length.
        if truncation and max_length is not None:
            sequences = [s[: max_length - int(add_special_tokens)] for s in sequences]
        encoded = [[self._token_to_id[c] for c in residues] for residues in sequences]
        if add_special_tokens:
            encoded = [[self.cls_token_id] + ids for ids in encoded]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class _FakeMaskedLM(nn.Module):
    """Returns the same logits vector at every position, regardless of input."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.arange(self.vocab_size, dtype=torch.float32)
        logits = logits.view(1, 1, -1).expand(batch, seq_len, -1)

        class _Output:
            pass

        out = _Output()
        out.logits = logits
        return out


def test_parse_mutant_single_and_multi():
    assert parse_mutant("A123T") == [("A", 123, "T")]
    assert parse_mutant("A123T:D45G") == [("A", 123, "T"), ("D", 45, "G")]


@pytest.mark.parametrize("mutant", ["", "A0T", "A12", "A12TT", "12T"])
def test_parse_mutant_rejects_invalid_substitution_notation(mutant):
    with pytest.raises(ValueError, match="Invalid substitution notation"):
        parse_mutant(mutant)


def test_optimal_window_matches_proteingym_get_optimal_window():
    # Whole sequence fits: no windowing.
    assert _optimal_window(position=5, full_len=10, window=20) == (0, 10)
    # Near the start: left-aligned window.
    assert _optimal_window(position=2, full_len=100, window=20) == (0, 20)
    # Near the end: right-aligned window.
    assert _optimal_window(position=98, full_len=100, window=20) == (80, 100)
    # Middle: centered window.
    assert _optimal_window(position=50, full_len=100, window=20) == (40, 60)


def test_load_assay_adds_unique_id_from_mutant_or_sequence(tmp_path):
    assay_path = tmp_path / "assay.csv"
    pl.DataFrame(
        {
            "mutated_sequence": ["ACDF", "ACDG"],
            "mutant": ["A1D", None],
            "DMS_score": [0.1, 0.2],
        }
    ).write_csv(assay_path)

    assay = _load_assay(tmp_path, "assay.csv", "assay-1")

    assert assay["id"].to_list() == ["assay-1:A1D", "assay-1:ACDG"]
    assert assay["mutated_sequence"].to_list() == ["ACDF", "ACDG"]
    assert assay["id"].n_unique() == assay.height

    indel_path = tmp_path / "indel.csv"
    pl.DataFrame({"mutated_sequence": ["ACD", "ACDE"]}).write_csv(indel_path)
    indels = _load_assay(tmp_path, "indel.csv", "indel-1")
    assert indels["id"].to_list() == ["indel-1:ACD", "indel-1:ACDE"]

    clinical_path = tmp_path / "clinical.csv"
    pl.DataFrame(
        {
            "mutated_sequence": ["ACD", "ACE"],
            "DMS_bin_score": ["Benign", "Pathogenic"],
        }
    ).write_csv(clinical_path)
    clinical = _load_assay(tmp_path, "clinical.csv", "clinical-1")
    assert clinical["dms_score_bin"].to_list() == ["Benign", "Pathogenic"]


def test_seeded_splits_are_reproducible_and_disjoint():
    frame = pl.DataFrame({"id": [f"variant-{index}" for index in range(10)]})

    first = _add_seeded_splits(frame, 0.8, 0.1, seed=1957723)
    second = _add_seeded_splits(frame, 0.8, 0.1, seed=1957723)

    assert first.equals(second)
    assert first["split"].value_counts().height == 3
    # Alphabetical sort of split names: test, train, validation.
    assert first.group_by("split").len().sort("split")["len"].to_list() == [1, 8, 1]


def test_seeded_splits_leave_tiny_assays_unchanged():
    frame = pl.DataFrame({"id": ["variant-0", "variant-1"]})

    assert _add_seeded_splits(frame, 0.8, 0.1, seed=1957723).equals(frame)


def test_proteingym_card_describes_track_layout_and_splits(tmp_path):
    config = type(
        "Config",
        (),
        {
            "base_url": "https://example.test/proteingym",
            "version": "v1.3",
            "reference_base_url": "https://example.test/reference_files",
            "track": "dms_substitutions",
            "seed": 7,
            "train_fraction": 0.8,
            "validation_fraction": 0.1,
            "create_split_subsets": True,
            "combine_assays": True,
        },
    )()
    combined = pl.DataFrame(
        {
            "id": ["assay-1:A1C"],
            "dms_id": ["assay-1"],
            "mutated_sequence": ["ACD"],
            "dms_score": [0.5],
            "fold_random_5": [0],
        }
    )
    reference = pl.DataFrame({"dms_id": ["assay-1"]})

    card_path = write_proteingym_card(
        tmp_path,
        config,
        _TRACKS["dms_substitutions"],
        combined,
        reference,
        ["train", "train_random", "train_stratified"],
    )

    card = card_path.read_text(encoding="utf-8")
    assert "ProteinGym `v1.3` DMS substitution assays" in card
    assert "DMS_substitutions.csv" in card
    assert "combined_<split>.parquet" in card
    assert "seed `7`" in card
    assert "`dms_score`" in card
    assert "`fold_random_5`" in card


def test_score_assay_masked_marginal_matches_closed_form():
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)
    wt_sequence = "ACDEFG"
    mutants = ["A1C", "C2A", "A1C:C2A"]

    scores = score_assay_masked_marginal(
        model, tokenizer, wt_sequence, mutants, device=torch.device("cpu"), batch_size=2
    )

    expected_log_probs = torch.log_softmax(
        torch.arange(tokenizer.vocab_size, dtype=torch.float32), dim=-1
    )

    def expected_score(mutant: str) -> float:
        total = 0.0
        for wt_aa, _, mut_aa in parse_mutant(mutant):
            total += (
                expected_log_probs[tokenizer.convert_tokens_to_ids(mut_aa)]
                - expected_log_probs[tokenizer.convert_tokens_to_ids(wt_aa)]
            ).item()
        return total

    expected = np.array([expected_score(m) for m in mutants])
    np.testing.assert_allclose(scores, expected, atol=1e-5)
    # Multi-mutant score is the sum of its constituent single-mutant scores.
    assert scores[2] == scores[0] + scores[1]


def test_score_assay_masked_marginal_windows_long_sequences_instead_of_truncating():
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)
    wt_sequence = "ACDEFG"

    scores = score_assay_masked_marginal(
        model,
        tokenizer,
        wt_sequence,
        ["A1C", "G6A"],
        device=torch.device("cpu"),
        batch_size=4,
        max_length=4,  # shorter than the full 7-token sequence: must window, not truncate
    )

    # Position 6 would fall outside a left-aligned truncation to 4 tokens, but
    # windowing (matching ProteinGym's get_optimal_window) still scores it.
    assert np.isfinite(scores[0])
    assert np.isfinite(scores[1])


def test_score_assay_masked_marginal_returns_nan_for_out_of_range_positions():
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)
    wt_sequence = "ACDEFG"

    scores = score_assay_masked_marginal(
        model,
        tokenizer,
        wt_sequence,
        ["A1C", "G100A"],
        device=torch.device("cpu"),
        batch_size=4,
    )

    assert np.isfinite(scores[0])
    assert np.isnan(scores[1])


def test_score_assay_masked_marginal_rejects_wild_type_mismatch():
    tokenizer = _FakeTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)

    with pytest.raises(ValueError, match="Wild-type residue mismatch"):
        score_assay_masked_marginal(
            model,
            tokenizer,
            "ACDEFG",
            ["C1A"],
            device=torch.device("cpu"),
        )


def test_compute_assay_spearman_perfect_correlation():
    model_scores = np.array([0.1, 0.5, 0.9, 0.2])
    dms_scores = np.array([1.0, 2.0, 3.0, 1.5])
    assert compute_assay_spearman(model_scores, dms_scores) == 1.0


def test_compute_assay_spearman_nan_with_too_few_valid_points():
    model_scores = np.array([np.nan, 0.5])
    dms_scores = np.array([1.0, 2.0])
    assert np.isnan(compute_assay_spearman(model_scores, dms_scores))


def test_aggregate_proteingym_scores_hierarchy():
    # Two proteins in "Activity" (assay-level scores 1.0/0.0 -> protein mean 0.5,
    # and a single 0.5 assay), one protein in "Binding" (mean 0.2).
    assay_results = pl.DataFrame(
        {
            "dms_id": ["a1", "a2", "a3", "b1"],
            "uniprot_id": ["P1", "P1", "P2", "P3"],
            "coarse_selection_type": ["Activity", "Activity", "Activity", "Binding"],
            "score": [1.0, 0.0, 0.5, 0.2],
        }
    )

    result = aggregate_proteingym_scores(assay_results)

    # Activity: P1 -> mean(1.0, 0.0) = 0.5, P2 -> 0.5 => category mean = 0.5
    assert result["per_category"]["Activity"] == 0.5
    # Binding: single protein, single assay.
    assert result["per_category"]["Binding"] == 0.2
    # Overall = mean across categories (each category weighted equally).
    assert result["overall"] == (0.5 + 0.2) / 2


def test_aggregate_proteingym_scores_uncertainty_is_zero_with_one_protein_per_category():
    # A single protein per category makes every bootstrap resample identical.
    assay_results = pl.DataFrame(
        {
            "dms_id": ["a1", "b1"],
            "uniprot_id": ["P1", "P3"],
            "coarse_selection_type": ["Activity", "Binding"],
            "score": [0.5, 0.2],
        }
    )

    result = aggregate_proteingym_scores(
        assay_results, compute_uncertainty=True, n_bootstrap=50
    )

    assert result["bootstrap_standard_error"] == pytest.approx(0.0)


def test_bootstrap_standard_error_is_deterministic_given_seed():
    assay_results = pl.DataFrame(
        {
            "dms_id": ["a1", "a2", "a3", "b1", "b2"],
            "uniprot_id": ["P1", "P2", "P4", "P3", "P5"],
            "coarse_selection_type": ["Activity"] * 3 + ["Binding"] * 2,
            "score": [1.0, 0.0, 0.5, 0.2, 0.8],
        }
    )

    first = bootstrap_standard_error_by_category(assay_results, n_resamples=200, seed=0)
    second = bootstrap_standard_error_by_category(assay_results, n_resamples=200, seed=0)

    assert first == second
    assert first >= 0.0


def test_score_assay_random_has_expected_length_and_seed_reproducibility():
    a = score_assay_random(10, seed=1)
    b = score_assay_random(10, seed=1)
    assert a.shape == (10,)
    np.testing.assert_array_equal(a, b)


def test_score_assay_blosum62_matches_manual_lookup():
    # A1C: substitution matrix score for Alanine->Cysteine is 0; A1D is -2.
    scores = score_assay_blosum62(["A1C", "A1D", "A1C:A1D"])
    np.testing.assert_allclose(scores, [0.0, -2.0, -2.0])


def test_official_cv_schemes_are_the_three_proteingym_folds():
    assert OFFICIAL_CV_SCHEMES == ("fold_random_5", "fold_modulo_5", "fold_contiguous_5")


def test_compute_assay_auc_perfect_separation():
    model_scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_assay_auc(model_scores, labels) == 1.0


def test_compute_assay_auc_nan_with_single_class():
    model_scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    assert np.isnan(compute_assay_auc(model_scores, labels))


def test_compute_assay_metric_dispatches_on_available_columns():
    model_scores = np.array([0.1, 0.5, 0.9])
    metric_name, _ = compute_assay_metric(model_scores, dms_scores=np.array([1.0, 2.0, 3.0]))
    assert metric_name == "spearman"

    metric_name, _ = compute_assay_metric(model_scores, dms_score_bin=np.array([0, 0, 1]))
    assert metric_name == "auc"


def test_compute_assay_ndcg_perfect_ranking_is_one():
    model_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    dms_scores = model_scores.copy()
    assert compute_assay_ndcg(model_scores, dms_scores) == pytest.approx(1.0)


def test_compute_assay_ndcg_nan_when_scores_are_constant():
    model_scores = np.array([0.1, 0.2, 0.3])
    dms_scores = np.array([1.0, 1.0, 1.0])
    assert np.isnan(compute_assay_ndcg(model_scores, dms_scores))


def test_compute_assay_top_recall_perfect_agreement():
    model_scores = np.arange(10, dtype=float)
    dms_scores = np.arange(10, dtype=float)
    assert compute_assay_top_recall(model_scores, dms_scores, top_true=10, top_model=10) == 1.0


def test_compute_assay_mcc_perfect_agreement():
    model_scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert compute_assay_mcc(model_scores, labels) == pytest.approx(1.0)


def test_compute_assay_mcc_nan_with_single_class():
    model_scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 1])
    assert np.isnan(compute_assay_mcc(model_scores, labels))


def test_compute_all_assay_metrics_computes_both_continuous_and_binary_metrics():
    model_scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_all_assay_metrics(
        model_scores,
        dms_scores=np.array([1.0, 2.0, 3.0, 4.0]),
        dms_score_bin=np.array([0, 0, 1, 1]),
    )
    assert set(metrics) == {"spearman", "ndcg", "top_recall", "auc", "mcc"}
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)


def test_aggregate_proteingym_metrics_skips_columns_that_are_entirely_null():
    assay_results = pl.DataFrame(
        {
            "coarse_selection_type": ["Activity", "Activity"],
            "uniprot_id": ["P1", "P2"],
            "spearman": [0.5, 0.9],
            "mcc": [None, None],
        }
    )
    summary = aggregate_proteingym_metrics(assay_results, ["spearman", "mcc", "auc"])
    assert set(summary) == {"spearman"}
    assert summary["spearman"]["overall"] == pytest.approx(0.7)


class _FakeBatchTokenizer:
    """Like _FakeTokenizer, but accepts a batch of sequences and pads them."""

    def __init__(self) -> None:
        self._token_to_id = {aa: i + 4 for i, aa in enumerate(_ALPHABET)}
        self.cls_token_id = 1
        self.mask_token_id = 2
        self.pad_token_id = 0
        self.vocab_size = len(self._token_to_id) + 4

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id[token]

    def __call__(
        self,
        sequences,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        if isinstance(sequences, str):
            sequences = [sequences]
        if truncation and max_length is not None:
            sequences = [s[: max_length - int(add_special_tokens)] for s in sequences]
        encoded = [[self._token_to_id[c] for c in s] for s in sequences]
        if add_special_tokens:
            encoded = [[self.cls_token_id] + ids for ids in encoded]
        width = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_score_assay_indel_matches_closed_form():
    tokenizer = _FakeBatchTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)
    wt_sequence = "ACDEFG"
    mutated_sequences = ["ACDEFGG", "ACDE"]  # an insertion and a deletion

    scores = score_assay_indel(
        model, tokenizer, wt_sequence, mutated_sequences, device=torch.device("cpu"), batch_size=2
    )

    expected_log_probs = torch.log_softmax(
        torch.arange(tokenizer.vocab_size, dtype=torch.float32), dim=-1
    )

    def expected_log_likelihood(sequence: str) -> float:
        ids = [tokenizer.cls_token_id] + [tokenizer.convert_tokens_to_ids(c) for c in sequence]
        return sum(expected_log_probs[i].item() for i in ids)

    expected = np.array(
        [expected_log_likelihood(s) - expected_log_likelihood(wt_sequence) for s in mutated_sequences]
    )
    np.testing.assert_allclose(scores, expected, atol=1e-4)


def test_score_assay_indel_excludes_cls_from_masked_likelihood():
    tokenizer = _FakeBatchTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)

    scores = score_assay_indel(
        model,
        tokenizer,
        "AC",
        ["ACD"],
        device=torch.device("cpu"),
        batch_size=2,
    )
    log_probs = torch.log_softmax(
        torch.arange(tokenizer.vocab_size, dtype=torch.float32), dim=-1
    )
    assert scores[0] == pytest.approx(log_probs[tokenizer.convert_tokens_to_ids("D")].item())


def test_score_assay_indel_truncates_residue_positions_with_input():
    tokenizer = _FakeBatchTokenizer()
    model = _FakeMaskedLM(tokenizer.vocab_size)
    wt_sequence = "A" * 40

    scores = score_assay_indel(
        model,
        tokenizer,
        wt_sequence,
        [wt_sequence + "A"],
        device=torch.device("cpu"),
        batch_size=64,
        max_length=32,
    )

    assert np.isfinite(scores[0])
    assert scores[0] == pytest.approx(0.0)


class _FakeEncoderModel(nn.Module):
    """Returns each token's id broadcast across a small hidden dimension."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = input_ids.unsqueeze(-1).float().expand(*input_ids.shape, self.hidden_size)

        class _Output:
            pass

        out = _Output()
        out.hidden_states = [hidden]
        return out


def test_pool_embeddings_mean_pools_valid_tokens_only():
    tokenizer = _FakeBatchTokenizer()
    model = _FakeEncoderModel(hidden_size=3)

    embeddings = pool_embeddings(
        model, tokenizer, ["AC", "ACD"], device=torch.device("cpu"), batch_size=2
    )

    # "AC" -> [CLS=1, A=4, C=5], mean = 10/3; padding must not affect the shorter row.
    assert embeddings.shape == (2, 3)
    np.testing.assert_allclose(embeddings[0], np.full(3, (1 + 4 + 5) / 3), atol=1e-5)


def test_resolve_folds_custom_kfold_partitions_all_variants():
    folds = resolve_folds(10, official_folds=None, strategy="custom_kfold", n_splits=5, seed=0)
    assert folds.shape == (10,)
    assert set(folds.tolist()) == set(range(5))


def test_resolve_folds_requires_official_column_when_not_custom():
    try:
        resolve_folds(10, official_folds=None, strategy="fold_random_5")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_assay_loop_uses_metric_override_from_score_assay(tmp_path):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)

    def score_assay(row, assay):
        # Real predictions would give a perfect (1.0) Spearman; the override
        # simulates a metric averaged across multiple CV schemes instead.
        return np.array([0.1, 0.2]), 0.42

    summary = proteingym_runner.run_assay_loop(
        data_dir, tmp_path / "out", None, score_assay
    )

    assert summary["per_category"]["Activity"] == pytest.approx(0.42)


def test_run_assay_loop_computes_bootstrap_uncertainty_when_requested(tmp_path):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)

    summary = proteingym_runner.run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        compute_uncertainty=True,
        n_bootstrap=20,
    )

    assert "bootstrap_standard_error" in summary


def test_run_assay_loop_extra_metrics_off_by_default(tmp_path):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)

    summary = proteingym_runner.run_assay_loop(
        data_dir, tmp_path / "out", None, lambda row, assay: np.array([0.5, 0.9])
    )

    assert "metrics" not in summary


def test_run_assay_loop_computes_extra_zero_shot_metrics_when_requested(tmp_path):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)

    summary = proteingym_runner.run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        compute_extra_zero_shot_metrics=True,
    )

    # dms_score_bin isn't present in _write_sourced_layout's assay, so only
    # the continuous-score metrics (not auc/mcc) should be aggregated.
    assert set(summary["metrics"]) == {"spearman", "ndcg", "top_recall"}


def test_run_assay_loop_pools_auc_across_genes_for_clinical_indels(tmp_path):
    # Matches performance_clinical_benchmarks.py's compute_pooled_auc: one AUC
    # over every gene's variants combined, not the average of per-gene AUCs.
    data_dir = tmp_path / "sourced"
    (data_dir / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "dms_id": ["gene-a", "gene-b"],
            "uniprot_id": ["gene-a", "gene-b"],
            "coarse_selection_type": ["Clinical", "Clinical"],
            "is_clinical": [True, True],
            "is_indel": [True, True],
        }
    ).write_parquet(data_dir / "reference.parquet")
    # Pathogenic=1, Benign=0 (pre-flip); gene-a's scores separate perfectly,
    # gene-b's are inverted, so the per-gene-average AUC would be (1.0+0.0)/2=0.5.
    pl.DataFrame(
        {"mutated_sequence": ["A", "B", "C", "D"], "dms_score_bin": [1, 0, 1, 0]}
    ).write_parquet(data_dir / "data" / "gene-a.parquet")
    pl.DataFrame(
        {"mutated_sequence": ["E", "F"], "dms_score_bin": [1, 0]}
    ).write_parquet(data_dir / "data" / "gene-b.parquet")

    model_scores = {
        "gene-a": np.array([0.2, 0.8, 0.3, 0.7]),
        "gene-b": np.array([0.6, 0.1]),
    }

    summary = proteingym_runner.run_assay_loop(
        data_dir, tmp_path / "out", None, lambda row, assay: model_scores[row["dms_id"]]
    )

    assert summary["aggregation"] == "pooled"
    assert summary["metric"] == "auc"
    # Pooled across both genes: 6 of 9 positive/negative pairs rank correctly.
    assert summary["overall"] == pytest.approx(6 / 9)


def test_run_cv_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 3))
    true_weights = np.array([1.0, -2.0, 0.5])
    targets = embeddings @ true_weights
    folds = resolve_folds(60, official_folds=None, strategy="custom_kfold", n_splits=5, seed=0)

    predictions = run_cv(embeddings, targets, folds, task="regression")

    assert not np.isnan(predictions).any()
    assert np.corrcoef(predictions, targets)[0, 1] > 0.99


def test_run_cv_supports_mlp_probe_type():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 3))
    targets = embeddings @ np.array([1.0, -2.0, 0.5])
    folds = resolve_folds(60, official_folds=None, strategy="custom_kfold", n_splits=5, seed=0)

    predictions = run_cv(embeddings, targets, folds, task="regression", probe_type="mlp")

    assert not np.isnan(predictions).any()


def test_run_cv_supports_torch_linear_probe_type():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 3))
    true_weights = np.array([1.0, -2.0, 0.5])
    targets = embeddings @ true_weights
    folds = resolve_folds(60, official_folds=None, strategy="custom_kfold", n_splits=5, seed=0)

    predictions = run_cv(
        embeddings,
        targets,
        folds,
        task="regression",
        probe_type="torch_linear",
        hyperparameters={"epochs": 500, "lr": 0.05},
    )

    assert not np.isnan(predictions).any()
    assert np.corrcoef(predictions, targets)[0, 1] > 0.99


def test_run_cv_search_space_tunes_probe_hyperparameters():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(80, 3))
    true_weights = np.array([1.0, -2.0, 0.5])
    targets = embeddings @ true_weights
    folds = resolve_folds(80, official_folds=None, strategy="custom_kfold", n_splits=5, seed=0)

    predictions = run_cv(
        embeddings,
        targets,
        folds,
        task="regression",
        probe_type="linear",
        search_space={"alpha": [0.001, 1.0, 100.0]},
    )

    assert not np.isnan(predictions).any()
    assert np.corrcoef(predictions, targets)[0, 1] > 0.99

    assert not np.isnan(predictions).any()


def _write_sourced_layout(root):
    (root / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "dms_id": ["assay-1"],
            "uniprot_id": ["P1"],
            "coarse_selection_type": ["Activity"],
        }
    ).write_parquet(root / "reference.parquet")
    pl.DataFrame({"mutated_sequence": ["ACD", "ACE"], "dms_score": [0.1, 0.2]}).write_parquet(
        root / "data" / "assay-1.parquet"
    )


def test_run_assay_loop_reads_assays_from_the_data_subfolder(tmp_path):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)
    output_dir = tmp_path / "out"

    summary = run_assay_loop(
        data_dir, output_dir, None, lambda row, assay: np.array([0.5, 0.9])
    )

    assert (output_dir / "assay-1_scored.parquet").exists()
    assert summary["per_category"]["Activity"] == pytest.approx(1.0)


def test_run_assay_loop_uses_local_data_without_hf_repo(tmp_path, monkeypatch):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)

    def fail_download(*args, **kwargs):
        raise AssertionError("HF should not be called without data_repo_id")

    monkeypatch.setattr(proteingym_runner, "download_from_hf", fail_download)

    summary = run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
    )

    assert summary["per_category"]["Activity"] == pytest.approx(1.0)


def test_run_assay_loop_uses_local_hf_snapshot_first(tmp_path, monkeypatch):
    data_dir = tmp_path / "downloaded"
    calls = []

    def fake_download(repo_id, local_dir, **kwargs):
        assert repo_id == "org/proteingym-dms_substitutions"
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        _write_sourced_layout(local_dir)
        return local_dir

    monkeypatch.setattr(proteingym_runner, "download_from_hf", fake_download)

    summary = run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        data_repo_id="org/proteingym-dms_substitutions",
    )

    assert summary["per_category"]["Activity"] == pytest.approx(1.0)
    assert len(calls) == 1


def test_run_assay_loop_downloads_after_local_hf_cache_miss(tmp_path, monkeypatch):
    data_dir = tmp_path / "downloaded"
    calls = []

    def fake_download(repo_id, local_dir, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise FileNotFoundError("HF snapshot is not cached")
        _write_sourced_layout(local_dir)
        return local_dir

    monkeypatch.setattr(proteingym_runner, "download_from_hf", fake_download)

    summary = run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        data_repo_id="org/proteingym-dms_substitutions",
    )

    assert summary["per_category"]["Activity"] == pytest.approx(1.0)
    assert [call.get("local_files_only", False) for call in calls] == [True, False]
    assert calls[1]["force_download"] is False


def test_run_assay_loop_falls_back_to_local_data_after_hf_failure(tmp_path, monkeypatch):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)
    calls = []

    def fake_download(repo_id, local_dir, **kwargs):
        calls.append(kwargs)
        raise ConnectionError("HF is unavailable")

    monkeypatch.setattr(proteingym_runner, "download_from_hf", fake_download)

    summary = run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        data_repo_id="org/proteingym-dms_substitutions",
    )

    assert summary["per_category"]["Activity"] == pytest.approx(1.0)
    assert len(calls) == 2


def test_run_assay_loop_force_download_skips_local_hf_snapshot(tmp_path, monkeypatch):
    data_dir = tmp_path / "sourced"
    _write_sourced_layout(data_dir)
    calls = []

    def fake_download(repo_id, local_dir, **kwargs):
        calls.append(kwargs)
        raise ConnectionError("HF is unavailable")

    monkeypatch.setattr(proteingym_runner, "download_from_hf", fake_download)

    summary = run_assay_loop(
        data_dir,
        tmp_path / "out",
        None,
        lambda row, assay: np.array([0.5, 0.9]),
        data_repo_id="org/proteingym-dms_substitutions",
        force_download=True,
    )

    assert summary["per_category"]["Activity"] == pytest.approx(1.0)
    assert len(calls) == 1
    assert calls[0]["force_download"] is True
    assert "local_files_only" not in calls[0]
