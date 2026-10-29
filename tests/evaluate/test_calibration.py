"""Calibration task: windowing, reference scoring, metrics, and sourcing."""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest
import torch
from datasets import Dataset, DatasetDict

from modules.evaluate.scripts.dataset_sourcing import run_hhblits
from modules.evaluate.scripts.dataset_sourcing.source_calibration import (
    SourceCalibrationConfig,
    build_calibration_frame,
    read_a3m_profiles,
)
from modules.evaluate.src.dataset.dataloader import tokenize_dataset
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.calibration import WINDOW_RESIDUES
from modules.evaluate.src.utils.amino_acids import CANONICAL_AMINO_ACIDS, resolve_aa_token_ids
from modules.evaluate.src.metrics.bootstrap import id_groups
from tests.evaluate.conftest import LONG_SEQUENCE

SEQUENCE = (LONG_SEQUENCE * 20)[:1200]
POSITIONS = [5, 600, 1190]


def _profile(position: int, consensus: str) -> list[float]:
    freqs = [0.0] * 20
    freqs[CANONICAL_AMINO_ACIDS.index(consensus)] = 1.0
    return [float(position), *freqs]


def _tokenize(
    tokenizer, max_length: int, random_truncate: bool = False, positions=POSITIONS
):
    raw = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "id": ["P1"],
                    "sequence": [SEQUENCE],
                    "targets": [[_profile(p, SEQUENCE[p]) for p in positions]],
                }
            )
        }
    )
    return tokenize_dataset(
        raw,
        tokenizer=tokenizer,
        task_type="calibration",
        sequence_column="sequence",
        label_column="targets",
        max_length=max_length,
        random_truncate=random_truncate,
    )["test"]


def test_windows_are_centred_and_clamped(protein_tokenizer):
    rows = _tokenize(protein_tokenizer, max_length=100_000)
    aa_ids = resolve_aa_token_ids(protein_tokenizer)
    residue_ids = protein_tokenizer(SEQUENCE, add_special_tokens=False)["input_ids"]

    assert rows["id"] == [f"P1:{p}" for p in POSITIONS]
    # Expected window starts: centred, then clamped at the N- and C-termini.
    expected_starts = [0, 600 - WINDOW_RESIDUES // 2, len(SEQUENCE) - WINDOW_RESIDUES]
    for row, pos, start in zip(rows, POSITIONS, expected_starts):
        input_ids, label = row["input_ids"], row["targets"]
        mask_idx, wt_idx = int(label[0]), int(label[1])
        assert len(input_ids) == WINDOW_RESIDUES + 2
        assert input_ids[0] == protein_tokenizer.bos_token_id
        assert input_ids[-1] == protein_tokenizer.eos_token_id
        assert input_ids[mask_idx] == protein_tokenizer.mask_token_id
        assert mask_idx == 1 + pos - start
        assert CANONICAL_AMINO_ACIDS[wt_idx] == SEQUENCE[pos]
        unmasked = list(input_ids)
        unmasked[mask_idx] = residue_ids[pos]
        assert unmasked[1:-1] == residue_ids[start : start + WINDOW_RESIDUES]
        assert [int(v) for v in label[2:22]] == list(aa_ids)


@pytest.mark.parametrize("random_truncate", [False, True])
def test_cropped_sequences_keep_only_in_window_positions(
    protein_tokenizer, random_truncate
):
    rows = _tokenize(
        protein_tokenizer,
        max_length=64,
        random_truncate=random_truncate,
        positions=list(range(0, len(SEQUENCE), 7)),
    )

    # 62 residues survive the crop, i.e. 8 or 9 of the every-7th positions.
    assert 8 <= len(rows) <= 9
    for row in rows:
        pos = int(row["id"].split(":")[1])
        assert CANONICAL_AMINO_ACIDS[int(row["targets"][1])] == SEQUENCE[pos]
    if not random_truncate:
        assert rows["id"] == [f"P1:{p}" for p in range(0, 62, 7)]


def _reference_softmax_over_aa(logits: np.ndarray, aa_ids: list[int]) -> np.ndarray:
    aa_logits = logits[aa_ids]
    exp = np.exp(aa_logits - aa_logits.max())
    return exp / exp.sum()


def _reference_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def test_scoring_matches_reference_formulas():
    task = get_task("calibration")
    rng = np.random.default_rng(0)
    batch, seq_len, vocab = 4, 6, 33
    logits = rng.normal(size=(batch, seq_len, vocab)).astype(np.float32)
    aa_ids = rng.permutation(vocab)[:20].tolist()
    freqs = rng.dirichlet(np.full(20, 0.3), size=batch)
    freqs[0] = 0.0
    freqs[0, 3] = 1.0  # exercise the zero-frequency clipping
    labels = [[i + 1, i, *aa_ids, *freqs[i]] for i in range(batch)]

    label_tensor = torch.tensor(labels, dtype=task.label_dtype)
    preds = task.extract_predictions(torch.from_numpy(logits), labels=label_tensor)
    rows = task.split_batch_rows(preds, label_tensor)

    for i, (pred, compact_label) in enumerate(rows):
        model_probs = _reference_softmax_over_aa(logits[i, i + 1], aa_ids)
        homolog = freqs[i] / freqs[i].sum()
        assert pred[0] == pytest.approx(_reference_kl(homolog, model_probs), rel=1e-4)
        assert pred[1] == int(np.argmax(model_probs))
        assert compact_label == [i, int(np.argmax(freqs[i]))]


def test_metrics_split_target_and_non_target_positions():
    task = get_task("calibration")
    # labels are [wt_idx, consensus_idx]; predictions are [kl, pred_idx].
    labels = [[0, 0], [1, 1], [2, 5], [3, 7]]
    predictions = [[0.1, 0], [0.3, 4], [1.0, 5], [2.0, 3]]

    scores = task.compute_metrics(predictions, labels, task.default_metrics())

    assert scores == pytest.approx(
        {
            "kl": 0.85,
            "kl_t": 0.2,
            "kl_nt": 1.5,
            "calibration": 100 * np.exp(-np.array([0.1, 0.3, 1.0, 2.0])).mean(),
            "calibration_t": 100 * np.exp(-np.array([0.1, 0.3])).mean(),
            "calibration_nt": 100 * np.exp(-np.array([1.0, 2.0])).mean(),
            "accuracy_t": 0.5,
            "accuracy_nt": 0.5,
            "memorization_nt": 0.5,
        }
    )


def test_bootstrap_resamples_whole_proteins():
    ids = ["P1:3", "P1:10", "P2:0", "sp|P3|X:5"]
    groups = id_groups([get_task("calibration").bootstrap_group_id(i) for i in ids])
    np.testing.assert_array_equal(groups, [0, 0, 1, 2])


def test_build_calibration_frame_filters_and_groups():
    proteome = pl.DataFrame(
        {"id": ["P1", "P2", "P3"], "sequence": ["ACDE", "MKLV", "WYXA"]}
    )

    def row(name, pos, wt, consensus, freq):
        freqs = {aa: 0.0 for aa in CANONICAL_AMINO_ACIDS}
        freqs[consensus] = freq
        freqs["A" if consensus != "A" else "C"] += 1.0 - freq
        return {"sequence_name": name, "position": pos, "wt_aa": wt, **freqs}

    consensus = pl.DataFrame(
        [
            row("P1", 2, "D", "D", 0.9),
            row("P1", 0, "A", "C", 0.8),
            row("P1", 1, "C", "C", 0.5),  # below threshold
            row("P2", 0, "M", "M", 0.9),
            row("P2", 1, "A", "A", 0.9),  # wt mismatch: drop all of P2
            row("P3", 2, "X", "A", 0.9),  # non-standard wt
            row("P3", 3, "A", "A", 0.9),
            row("P9", 0, "A", "A", 0.9),  # not in the proteome
        ]
    )

    frame = build_calibration_frame(proteome, consensus, 0.7)

    assert frame["id"].to_list() == ["P1", "P3"]
    assert frame["sequence"].to_list() == ["ACDE", "WYXA"]
    p1_targets = frame["targets"][0].to_list()
    assert [t[0] for t in p1_targets] == [0.0, 2.0]
    assert all(len(t) == 21 for t in p1_targets)
    assert p1_targets[0][1 + CANONICAL_AMINO_ACIDS.index("C")] == pytest.approx(0.8)
    assert [t[0] for t in frame["targets"][1].to_list()] == [3.0]


A3M = """>Q1 query
ACDEF
>h1
ACkkDEF
>h2
AC-EW
>h3
WX-EW
"""


def test_read_a3m_profiles_counts_query_and_homologs(tmp_path):
    (tmp_path / "Q1.a3m").write_text(A3M)
    (tmp_path / "Q2.align.a3m").write_text(">Q2\nMK\n>h1\nMK\n")

    proteome, consensus = read_a3m_profiles(tmp_path, 0.7, min_homologs=2)

    assert proteome.to_dicts() == [{"id": "Q1", "sequence": "ACDEF"}]
    # Insertions are dropped, gaps and X are not counted, the query is.
    # Column 4 (F/F/W/W) is below the threshold.
    assert consensus["position"].to_list() == [0, 1, 2, 3]
    assert consensus["wt_aa"].to_list() == ["A", "C", "D", "E"]
    assert consensus["A"][0] == pytest.approx(0.75)
    assert consensus["W"][0] == pytest.approx(0.25)
    assert consensus["C"][1] == pytest.approx(1.0)
    assert consensus["D"][2] == pytest.approx(1.0)

    proteome, _ = read_a3m_profiles(tmp_path, 0.7, min_homologs=1)
    assert proteome["id"].to_list() == ["Q1", "Q2"]


def test_source_config_requires_one_input(tmp_path):
    SourceCalibrationConfig(a3m_dir=tmp_path)
    SourceCalibrationConfig(proteome_path=tmp_path, consensus_path=tmp_path)
    for kwargs in (
        {},
        {"proteome_path": tmp_path},
        {"a3m_dir": tmp_path, "proteome_path": tmp_path, "consensus_path": tmp_path},
    ):
        with pytest.raises(ValueError, match="a3m_dir"):
            SourceCalibrationConfig(**kwargs)


def test_run_hhblits_shards_resumes_and_feeds_sourcing(tmp_path):
    fake = tmp_path / "hhblits"
    calls = tmp_path / "calls.txt"
    fake.write_text(
        f"""#!{sys.executable}
import sys
args = sys.argv[1:]
opt = lambda flag: args[args.index(flag) + 1]
open({str(calls)!r}, "a").write(" ".join(args) + "\\n")
query = open(opt("-i")).read()
open(opt("-oa3m"), "w").write(query + ">hom\\n" + query.split()[-1] + "\\n")
"""
    )
    fake.chmod(0o755)
    fasta = tmp_path / "proteome.fasta"
    fasta.write_text(">sp|P1|A_HUMAN x\nMKV\n>sp|P2|B_HUMAN\nAC\nDE\n>P3\nWY\n")
    config = run_hhblits.RunHHblitsConfig(
        fasta=str(fasta),
        database=tmp_path / "db",
        work_dir=tmp_path / "work",
        hhblits_bin=str(fake),
        shard_index=0,
        shard_count=2,
    )
    a3m_dir = config.work_dir / "a3m"
    a3m_dir.mkdir(parents=True)
    (a3m_dir / "P3.a3m").write_text(">P3\nWY\n")  # already aligned

    run_hhblits.run(config)

    # Shard 0 of 2 is P1 and P3; P3 exists, so only P1 is aligned.
    assert sorted(p.name for p in a3m_dir.glob("*.a3m")) == ["P1.a3m", "P3.a3m"]
    call = calls.read_text().splitlines()
    assert len(call) == 1
    assert f"-d {tmp_path / 'db'}" in call[0]
    assert "-cpu 4 -id 99 -cov 50 -qid 50" in call[0]

    proteome, _ = read_a3m_profiles(a3m_dir, 0.7, min_homologs=1)
    assert proteome.to_dicts() == [{"id": "P1", "sequence": "MKV"}]

    assert run_hhblits.read_fasta(fasta)["P2"] == "ACDE"
