"""Tests for the contact_prediction task: label alignment/collation,
per-sequence row extraction, metrics, and the bilinear contact model.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from transformers import EsmConfig

from modules.evaluate.src.metrics import contact_metrics
from modules.evaluate.src.model.contact_model import (
    ContactModelConfig,
    ContactPredictionModel,
)
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.contact_prediction import IGNORE_INDEX
from modules.evaluate.src.utils.controls import ControlBuilder
from modules.evaluate.src.utils.inference import pad_for_gather


def _task():
    return get_task("contact_prediction")


def test_align_labels_crops_and_shifts_pairs():
    task = _task()
    # Residues 0..9; keep a window of 4 residues starting at residue 2
    # (offset=1 for a single leading BOS token).
    label = [[0, 1], [2, 3], [4, 5], [2, 4], [8, 9]]
    aligned = task.align_labels(
        label, num_residues=10, start=2, kept=4, offset=1, input_len=6, example_index=0
    )
    # Only pairs fully inside [2, 6) survive: (2,3)->(1,2)->offset shift, (4,5), (2,4)
    assert aligned["res_start"] == 1
    assert aligned["res_end"] == 5
    assert sorted(aligned["pairs"]) == sorted(
        [
            [1 + (2 - 2), 1 + (3 - 2)],  # [1, 2]
            [1 + (4 - 2), 1 + (5 - 2)],  # [3, 4]
            [1 + (2 - 2), 1 + (4 - 2)],  # [1, 3]
        ]
    )


def test_collate_labels_is_symmetric_and_masks_short_range_and_padding():
    task = _task()
    seq_len = 8
    labels = [
        # full-length sequence (no special tokens), one contact
        {"pairs": [[0, 7]], "res_start": 0, "res_end": 8},
        # short sequence + a sub-min_sep pair
        {"pairs": [[0, 1]], "res_start": 0, "res_end": 4},
    ]
    collated = task.collate_labels(labels, batch_size=2, seq_len=seq_len)

    assert collated.shape == (2, seq_len, seq_len)
    assert collated.dtype == task.label_dtype
    # Symmetric positive contact.
    assert collated[0, 0, 7] == 1
    assert collated[0, 7, 0] == 1
    # Valid (sep >= MIN_SEP) but non-contact pair is an explicit negative.
    assert collated[0, 0, 6] == 0
    # Below MIN_SEP: always ignored, even on the diagonal.
    assert collated[0, 0, 1] == IGNORE_INDEX
    # Second example: res_end=4, so anything touching padding (>=4) is ignored...
    assert collated[1, 0, 5] == IGNORE_INDEX
    # ...and the listed pair [0, 1] has sep=1 < MIN_SEP, so it stays ignored
    # rather than being (incorrectly) marked a positive.
    assert collated[1, 0, 1] == IGNORE_INDEX


def test_collate_labels_excludes_special_token_positions():
    task = _task()
    seq_len = 10
    # offset=1 leading CLS, 8 residues, 1 trailing EOS: real residues are
    # indices [1, 9). Index 0 (CLS) and 9 (EOS) must never be scored, even
    # though they fall at sep >= MIN_SEP from other positions.
    labels = [{"pairs": [], "res_start": 1, "res_end": 9}]
    collated = task.collate_labels(labels, batch_size=1, seq_len=seq_len)

    assert (collated[0, 0, :] == IGNORE_INDEX).all()
    assert (collated[0, :, 0] == IGNORE_INDEX).all()
    assert (collated[0, 9, :] == IGNORE_INDEX).all()
    assert (collated[0, :, 9] == IGNORE_INDEX).all()
    # A genuine residue pair at valid separation is still scored.
    assert collated[0, 1, 8] == 0


def test_split_batch_rows_returns_explicit_metric_metadata():
    task = _task()
    # Long enough that every residue has *some* partner at sep >= MIN_SEP, so
    # the reconstructed length sentinel exactly equals seq_len.
    seq_len = 2 * task.MIN_SEP + 2
    # Build a realistic dense label matrix (whole valid region filled, not
    # just the listed positives) via collate_labels itself.
    labels = task.collate_labels(
        [{"pairs": [[0, seq_len - 1]], "res_start": 0, "res_end": seq_len}],
        batch_size=1,
        seq_len=seq_len,
    )
    preds = torch.rand(1, seq_len, seq_len)

    rows = task.split_batch_rows(preds, labels)
    assert len(rows) == 1
    pred_row, label_row = rows[0]
    assert set(pred_row) == {"scores", "separations"}
    assert set(label_row) == {"length", "values"}
    assert label_row["length"] == seq_len
    # One positive pair, counted once (not twice, despite symmetry).
    assert label_row["values"].count(1) == 1
    positive_index = label_row["values"].index(1)
    assert pred_row["separations"][positive_index] == seq_len - 1


def test_split_batch_rows_length_uses_span_not_scored_count():
    task = _task()
    # An 8-residue chain: the two edge residues each have a scored partner
    # (sep=7 >= MIN_SEP=6), but every middle residue's partners are all
    # closer than MIN_SEP, so they have zero scored pairs of their own.
    # Counting scored columns would undercount L (4 instead of 8); the real
    # span (min..max scored index) must still recover the true length.
    seq_len = 8
    labels = task.collate_labels(
        [{"pairs": [], "res_start": 0, "res_end": seq_len}],
        batch_size=1,
        seq_len=seq_len,
    )
    scored_cols = (labels[0] != IGNORE_INDEX).any(dim=0)
    assert scored_cols.sum().item() < seq_len  # middle residues have no partner

    preds = torch.rand(1, seq_len, seq_len)
    rows = task.split_batch_rows(preds, labels)
    _, label_row = rows[0]
    assert label_row["length"] == seq_len


def test_pad_for_gather_pads_both_contact_map_axes():
    class TwoProcessAccelerator:
        num_processes = 2

        def __init__(self):
            self.dims: list[int] = []

        def pad_across_processes(self, tensor, *, dim, pad_index):
            self.dims.append(dim)
            return tensor

    accelerator = TwoProcessAccelerator()
    contact_map = torch.zeros((2, 8, 8))

    result = pad_for_gather(contact_map, accelerator, IGNORE_INDEX, dims=(1, 2))

    assert result is contact_map
    assert accelerator.dims == [1, 2]


def test_scramble_labels_preserves_length_and_shuffles_values():
    task = _task()
    rng = np.random.default_rng(0)
    labels = [{"length": 6, "values": [1, 0, 0]}, {"length": 4, "values": [1]}]
    scrambled = task.scramble_labels(labels, rng)
    assert [row["length"] for row in scrambled] == [6, 4]
    assert sorted(scrambled[0]["values"]) == sorted(labels[0]["values"])
    assert sorted(scrambled[1]["values"]) == sorted(labels[1]["values"])


def test_precision_at_l_ranks_within_bin_and_uses_length():
    # length=10 -> top-L/5 = top-2 candidates in the "long" bin (sep>=24).
    predictions = [{"scores": [0.9, 0.1, 0.8], "separations": [30, 25, 26]}]
    labels = [{"length": 10, "values": [1, 0, 0]}]
    score = contact_metrics.precision_at_l(
        predictions, labels, "macro", min_sep=24, max_sep=None, fraction=5
    )
    # top-2 by score: 0.9 (label 1), 0.8 (label 0) -> precision 0.5
    assert score == 0.5


def test_contact_range_auc_averages_decile_precisions_per_sequence():
    # length=10, sep>=24 ("long" bin): 4 candidates ranked by score
    # (0.9->1, 0.8->0, 0.7->1, 0.1->0). Golden value for the mean, over the
    # 10 deciles of L, of precision among the top round(decile * L) of them.
    predictions = [
        {"scores": [0.9, 0.1, 0.8, 0.7], "separations": [30, 25, 26, 28]}
    ]
    labels = [{"length": 10, "values": [1, 0, 1, 0]}]
    auc = contact_metrics.contact_range_auc(
        predictions, labels, "macro", min_sep=24, max_sep=None
    )
    assert auc == pytest.approx(0.6166666666666666)


def _tiny_esm_config() -> EsmConfig:
    return EsmConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        vocab_size=25,
        pad_token_id=1,
        max_position_embeddings=64,
    )


def test_contact_prediction_model_forward_is_symmetric_and_trains():
    config = ContactModelConfig(_tiny_esm_config(), rank=4)
    model = ContactPredictionModel(config)
    assert model.base_model_prefix == "trunk"

    input_ids = torch.randint(4, 20, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    labels = torch.full((2, 6, 6), IGNORE_INDEX, dtype=torch.long)
    labels[:, 0, 3] = 1
    labels[:, 3, 0] = 1
    labels[:, 1, 4] = 0
    labels[:, 4, 1] = 0

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert outputs.logits.shape == (2, 6, 6)
    assert torch.allclose(outputs.logits, outputs.logits.transpose(1, 2), atol=1e-5)
    assert outputs.loss is not None
    outputs.loss.backward()
    assert model.query.weight.grad is not None


def test_build_model_uses_default_rank_and_honors_model_kwargs_override(tmp_path):
    from transformers import EsmModel

    trunk_dir = tmp_path / "tiny-trunk"
    EsmModel(_tiny_esm_config()).save_pretrained(trunk_dir)

    task = _task()
    default_model = task.build_model(str(trunk_dir), num_labels=2)
    assert default_model.query.out_features == task.RANK
    assert default_model.config.pos_weight == task.POS_WEIGHT

    overridden_model = task.build_model(
        str(trunk_dir), num_labels=2, model_kwargs={"rank": 4, "pos_weight": 5.0}
    )
    assert overridden_model.query.out_features == 4
    assert overridden_model.config.pos_weight == 5.0


def test_compute_separation_prior_matches_empirical_rate():
    labels = torch.full((1, 5, 5), IGNORE_INDEX, dtype=torch.long)
    # sep=2 pairs: one positive, one negative -> prior[2] == 0.5
    labels[0, 0, 2] = 1
    labels[0, 2, 0] = 1
    labels[0, 1, 3] = 0
    labels[0, 3, 1] = 0
    dataloader = [{"labels": labels}]

    prior = ControlBuilder._compute_separation_prior(dataloader, max_sep=4)
    assert prior[2] == 0.5


def test_build_separation_prior_control_supports_dict_pairwise_rows():
    labels = torch.full((1, 5, 5), IGNORE_INDEX, dtype=torch.long)
    labels[0, 0, 2] = 1
    labels[0, 2, 0] = 1
    labels[0, 1, 3] = 0
    labels[0, 3, 1] = 0

    class _Prepared:
        train_dataloader = [{"labels": labels}]

    class _Workspace:
        task_type = "categorical_jacobian"

    predictions_df = pl.DataFrame(
        {
            "id": ["example-1"],
            "prediction": pl.Series(
                "prediction",
                [{"scores": [0.9, 0.1], "separations": [2, 2]}],
                strict=False,
            ),
        }
    )

    result = ControlBuilder._build_separation_prior_control(
        _Workspace(), _Prepared(), predictions_df
    )
    row = result["prediction"].to_list()[0]
    assert row["separations"] == [2, 2]
    assert row["scores"] == pytest.approx([0.5, 0.5])
