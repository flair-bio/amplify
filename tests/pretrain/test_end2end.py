"""End-to-end tests for the pretraining pipeline.

These run the *real* components (dataset -> dataloader -> collator -> model ->
optimizer -> scheduler -> Trainer) against a tiny synthetic Parquet corpus, so
they catch wiring breakages that unit tests with mocks cannot. Component-level
behaviour is tested in ``test_dataset.py``, ``test_dataloader.py``,
``test_collator.py`` and ``test_trainer.py``; this file only covers what
requires the whole stack assembled.

The config is built inline rather than loaded from YAML so the test is
self-contained: everything it depends on is visible in this file.

Everything is forced onto CPU and kept to a handful of steps, so the module
runs in a few seconds. The synthetic corpus deliberately uses the same column
names as the real assembled output of ``modules/data``
(``cluster_rep_at_30`` / ``red_score``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets.table import MemoryMappedTable
from pydantic import ValidationError
from torch.utils.data import ConcatDataset

from modules.pretrain.src.dataset.collator import CollatorConfig, get_collator
from modules.pretrain.src.dataset.dataloader import (
    DataLoaderConfig,
    build_split_dataloader,
)
from modules.pretrain.src.dataset.dataset import (
    DatasetConfig,
    DataSourceConfig,
    EpochPlan,
    build_epoch_plan,
    prepare_sources,
)

AMINO_ACIDS = "LAGVSERTIDPKQNFYMHWC"
AMBIGUOUS = ["<unk>", "X", "B", "O", "U", "Z", "J"]
# 5 specials + separator + 6 ambiguous residues + 20 amino acids = 32.
VOCAB = ["<pad>", "<unk>", "<mask>", "<bos>", "<eos>", "|"] + AMBIGUOUS[1:] + list(
    AMINO_ACIDS
)


def _concat(prepared, split="train"):
    """Build the persistent ConcatDataset for a split, mirroring run_pretrain.py."""
    attr = "train_dataset" if split == "train" else "val_dataset"
    return ConcatDataset([getattr(p, attr) for p in prepared])


def _tokenizer_config():
    from modules.pretrain.src.model.tokenizer import TokenizerConfig

    return TokenizerConfig(
        vocab=VOCAB,
        vocab_size=len(VOCAB),
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask>",
        bos_token="<bos>",
        eos_token="<eos>",
        ambiguous_tokens=AMBIGUOUS,
        remove_ambiguous=True,
    )


def _collator_config(**overrides):
    return CollatorConfig(
        **{
            "mlm": True,
            "mlm_probability": 0.15,
            "masking_type": "fixed",
            "masking_k": 10,
            "max_length": 64,
            "packing": False,
            "random_truncate": True,
            "exclude_special_tokens_from_masking": True,
            **overrides,
        }
    )


def _write_corpus(directory, num_rows: int, seed: int):
    """Write a tiny Parquet shard shaped like the real assembled data."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)

    lengths = rng.integers(20, 80, size=num_rows)
    sequences = ["".join(rng.choice(list(AMINO_ACIDS), size=int(n))) for n in lengths]
    # ~3 rows per cluster, so per-cluster deduplication actually does something.
    clusters = [f"c{i // 3:05d}" for i in range(num_rows)]

    pq.write_table(
        pa.table(
            {
                "sequence": sequences,
                "sequence_length": lengths.astype(np.uint32),
                "cluster_rep_at_30": clusters,
                "red_score": rng.uniform(0.0, 1.0, size=num_rows).astype(np.float32),
            }
        ),
        directory / "part-000.parquet",
    )
    return directory


@pytest.fixture
def corpus(tmp_path):
    """Two small sources, mirroring a multi-dataset mixture."""
    _write_corpus(tmp_path / "src_a", num_rows=300, seed=0)
    _write_corpus(tmp_path / "src_b", num_rows=300, seed=1)
    return tmp_path


@pytest.fixture
def dataset_config(corpus):
    return DatasetConfig(
        sources={
            "src_a": {
                "type": "parquet",
                "path": str(corpus / "src_a"),
                "cluster_column": "cluster_rep_at_30",
            },
            "src_b": {
                "type": "parquet",
                "path": str(corpus / "src_b"),
                "cluster_column": "cluster_rep_at_30",
            },
        },
        score_column="red_score",
        score_curriculum={
            "type": "linear",
            "start_score": 0.0,
            "end_score": 0.5,
            "ramp_epochs": 2,
        },
        val_min_score=0.0,
        val_holdout_modulus=5,
        seed=7,
    )


class TestPrepareSources:
    """Source loading across a multi-dataset mixture."""

    def test_zero_fraction_source_is_skipped_without_touching_its_path(
        self, dataset_config, tmp_path
    ):
        # Point the disabled source at a path that does not exist: if
        # `prepare_sources` tried to load it anyway, this would raise.
        dataset_config.sources["src_b"] = dataset_config.sources["src_b"].model_copy(
            update={
                "path": str(tmp_path / "does_not_exist"),
                "sampling_fraction": 0.0,
            }
        )
        prepared = prepare_sources(dataset_config)
        assert [p.name for p in prepared] == ["src_a"]

    def test_all_sources_zero_fraction_raises(self, dataset_config):
        for name, source in dataset_config.sources.items():
            dataset_config.sources[name] = source.model_copy(
                update={"sampling_fraction": 0.0}
            )
        with pytest.raises(ValueError, match="sampling_fraction"):
            prepare_sources(dataset_config)

    def test_hub_source_loads(self, dataset_config, corpus):
        # `load_dataset` accepts a local directory as the repo id, so the
        # `type="hub"` branch runs for real without touching the network.
        dataset_config.sources["src_b"] = DataSourceConfig(
            type="hub",
            repo_id=str(corpus / "src_b"),
            cluster_column="cluster_rep_at_30",
        )
        prepared = prepare_sources(dataset_config)
        assert [p.name for p in prepared] == ["src_a", "src_b"]
        assert len(prepared[1].train_dataset) > 0

    def test_hub_source_requires_repo_id(self):
        with pytest.raises(ValidationError, match="repo_id"):
            DataSourceConfig(type="hub", cluster_column="cluster_rep_at_30")


def _write_multi_threshold_corpus(directory, num_rows: int = 300, seed: int = 0):
    """Corpus carrying nested clusterings at 30/50/70% identity."""
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    lengths = rng.integers(20, 80, size=num_rows)
    sequences = ["".join(rng.choice(list(AMINO_ACIDS), size=int(n))) for n in lengths]
    pq.write_table(
        pa.table(
            {
                "sequence": sequences,
                "sequence_length": lengths.astype(np.uint32),
                # Nested: each 30% cluster splits into finer 50% then 70% ones.
                "cluster_rep_at_30": [f"c30_{i // 12:04d}" for i in range(num_rows)],
                "cluster_rep_at_50": [f"c50_{i // 6:04d}" for i in range(num_rows)],
                "cluster_rep_at_70": [f"c70_{i // 3:04d}" for i in range(num_rows)],
                "red_score": rng.uniform(0.0, 1.0, size=num_rows).astype(np.float32),
            }
        ),
        directory / "part-000.parquet",
    )
    return directory


class TestClusterColumnReuse:
    """split_column / extra_cluster_columns keep the split cache threshold-agnostic."""

    ALL_THRESHOLDS = [
        "cluster_rep_at_30",
        "cluster_rep_at_50",
        "cluster_rep_at_70",
    ]

    def _config(self, path, cluster_column, **source_overrides):
        return DatasetConfig(
            sources={
                "src": {
                    "type": "parquet",
                    "path": str(path),
                    "cluster_column": cluster_column,
                    **source_overrides,
                }
            },
            score_column="red_score",
            score_curriculum={
                "type": "linear",
                "start_score": 0.0,
                "end_score": 0.5,
                "ramp_epochs": 2,
            },
            val_min_score=0.0,
            val_holdout_modulus=5,
            seed=7,
        )

    @pytest.fixture
    def corpus_dir(self, tmp_path):
        return _write_multi_threshold_corpus(tmp_path / "multi")

    @staticmethod
    def _cached_columns(dataset):
        """Columns physically carried in the split cache, behind the projection."""
        return MemoryMappedTable.from_file(
            dataset.cache_files[0]["filename"]
        ).column_names

    @staticmethod
    def _cached_chunks(dataset):
        """Arrow record batches in the split cache (one metadata read each)."""
        return MemoryMappedTable.from_file(
            dataset.cache_files[0]["filename"]
        ).table.column(0).num_chunks

    def test_split_cache_honours_writer_batch_size(self, tmp_path):
        # `datasets` would emit one record batch per 1000 rows; opening such a
        # file costs a metadata read per batch, which dominates startup on a
        # network filesystem at billion-row scale.
        corpus = _write_multi_threshold_corpus(tmp_path / "chunks", num_rows=4000)
        config = self._config(corpus, "cluster_rep_at_30")
        config.split_writer_batch_size = 4000
        prepared = prepare_sources(config)[0]
        assert self._cached_chunks(prepared.train_dataset) == 1

    def test_training_dataset_is_projected_to_sequence(self, corpus_dir):
        # Training reads only `sequence`; carrying the cluster columns into the
        # hot path costs one extra buffer read per row in `__getitem__`.
        prepared = prepare_sources(
            self._config(
                corpus_dir,
                "cluster_rep_at_70",
                split_column="cluster_rep_at_30",
                extra_cluster_columns=self.ALL_THRESHOLDS,
            )
        )[0]
        assert prepared.train_dataset.column_names == ["sequence"]
        assert prepared.val_dataset.column_names == ["sequence"]
        # Projection is a view: rows and order are untouched.
        assert len(prepared.train_dataset) == prepared.train_index.order.size

    def test_defaults_match_legacy_single_column_behaviour(self, corpus_dir):
        # Unset split_column/extra_cluster_columns must reproduce the old
        # projection and holdout exactly, so caches built before this change
        # still hit.
        prepared = prepare_sources(self._config(corpus_dir, "cluster_rep_at_50"))[0]
        assert self._cached_columns(prepared.train_dataset) == [
            "sequence",
            "red_score",
            "cluster_rep_at_50",
            "sequence_length",
        ]

    def test_split_is_identical_across_thresholds(self, corpus_dir):
        """The holdout follows split_column, not cluster_column."""
        fingerprints, val_rows = set(), set()
        for threshold in self.ALL_THRESHOLDS:
            prepared = prepare_sources(
                self._config(
                    corpus_dir,
                    threshold,
                    split_column="cluster_rep_at_30",
                    extra_cluster_columns=self.ALL_THRESHOLDS,
                )
            )[0]
            fingerprints.add(prepared.train_dataset._fingerprint)
            val_rows.add(tuple(prepared.val_dataset["sequence"]))

        assert len(fingerprints) == 1, "split+flatten cache must be shared"
        assert len(val_rows) == 1, "val split must be comparable across thresholds"

    def test_split_differs_across_thresholds_without_split_column(self, corpus_dir):
        """Guards the problem being fixed: today's behaviour reshuffles the holdout."""
        fingerprints = {
            prepare_sources(self._config(corpus_dir, t))[0].train_dataset._fingerprint
            for t in self.ALL_THRESHOLDS
        }
        assert len(fingerprints) == len(self.ALL_THRESHOLDS)

    def test_cluster_index_still_tracks_cluster_column(self, corpus_dir):
        """Shared split cache, but grouping must still follow cluster_column."""
        cluster_counts = {}
        for threshold in self.ALL_THRESHOLDS:
            prepared = prepare_sources(
                self._config(
                    corpus_dir,
                    threshold,
                    split_column="cluster_rep_at_30",
                    extra_cluster_columns=self.ALL_THRESHOLDS,
                )
            )[0]
            cluster_counts[threshold] = prepared.train_index.num_clusters

        assert (
            cluster_counts["cluster_rep_at_30"]
            < cluster_counts["cluster_rep_at_50"]
            < cluster_counts["cluster_rep_at_70"]
        )

    def test_extra_columns_are_carried_and_sorted(self, corpus_dir):
        prepared = prepare_sources(
            self._config(
                corpus_dir,
                "cluster_rep_at_70",
                split_column="cluster_rep_at_30",
                extra_cluster_columns=self.ALL_THRESHOLDS,
            )
        )[0]
        assert self._cached_columns(prepared.train_dataset) == [
            "sequence",
            "red_score",
            *self.ALL_THRESHOLDS,
            "sequence_length",
        ]

    def test_missing_split_column_raises(self, corpus_dir):
        with pytest.raises(KeyError, match="Split column 'nope'"):
            prepare_sources(
                self._config(corpus_dir, "cluster_rep_at_30", split_column="nope")
            )

    def test_missing_extra_cluster_column_raises(self, corpus_dir):
        with pytest.raises(KeyError, match="Extra cluster column 'nope'"):
            prepare_sources(
                self._config(
                    corpus_dir, "cluster_rep_at_30", extra_cluster_columns=["nope"]
                )
            )


class TestSplitFingerprintStability:
    """The manual `_split_fingerprint` must ignore layout-only knobs but stay
    sensitive to anything that changes which rows end up in the split.

    See `modules/pretrain/fingerprint_stabilization_plan.md`. Without this,
    tuning `dataset.num_proc` or `dataset.split_writer_batch_size` forces a
    full re-filter+re-flatten even though neither changes a single output
    row (confirmed against `datasets==4.8.5`'s own fingerprinting, which
    hashes both).
    """

    def _config(self, path, **source_overrides):
        return DatasetConfig(
            sources={
                "src": {
                    "type": "parquet",
                    "path": str(path),
                    "cluster_column": "cluster_rep_at_30",
                    **source_overrides,
                }
            },
            score_column="red_score",
            score_curriculum={
                "type": "linear",
                "start_score": 0.0,
                "end_score": 0.5,
                "ramp_epochs": 2,
            },
            val_min_score=0.0,
            val_holdout_modulus=5,
            seed=7,
        )

    @pytest.fixture
    def corpus_dir(self, tmp_path):
        return _write_multi_threshold_corpus(tmp_path / "fp", num_rows=300)

    def test_stable_across_num_proc(self, corpus_dir):
        config_a = self._config(corpus_dir)
        config_a.num_proc = 1
        config_b = self._config(corpus_dir)
        config_b.num_proc = 2
        fp_a = prepare_sources(config_a)[0].train_dataset._fingerprint
        fp_b = prepare_sources(config_b)[0].train_dataset._fingerprint
        assert fp_a == fp_b

    def test_stable_across_split_writer_batch_size(self, corpus_dir):
        config_a = self._config(corpus_dir)
        config_a.split_writer_batch_size = 50
        config_b = self._config(corpus_dir)
        config_b.split_writer_batch_size = 300
        fp_a = prepare_sources(config_a)[0].train_dataset._fingerprint
        fp_b = prepare_sources(config_b)[0].train_dataset._fingerprint
        assert fp_a == fp_b

    def test_train_and_val_fingerprints_never_collide(self, corpus_dir):
        prepared = prepare_sources(self._config(corpus_dir))[0]
        assert prepared.train_dataset._fingerprint != prepared.val_dataset._fingerprint

    def test_changes_on_split_column(self, corpus_dir):
        fp_a = prepare_sources(
            self._config(corpus_dir, split_column="cluster_rep_at_30")
        )[0].train_dataset._fingerprint
        fp_b = prepare_sources(
            self._config(corpus_dir, split_column="cluster_rep_at_50")
        )[0].train_dataset._fingerprint
        assert fp_a != fp_b

    def test_changes_on_val_holdout_modulus(self, corpus_dir):
        config_a = self._config(corpus_dir)
        config_b = self._config(corpus_dir)
        config_b.val_holdout_modulus = 7
        fp_a = prepare_sources(config_a)[0].train_dataset._fingerprint
        fp_b = prepare_sources(config_b)[0].train_dataset._fingerprint
        assert fp_a != fp_b

    def test_changes_on_source_file_mtime(self, corpus_dir):
        config = self._config(corpus_dir)
        fp_before = prepare_sources(config)[0].train_dataset._fingerprint

        parquet_file = corpus_dir / "part-000.parquet"
        new_mtime = parquet_file.stat().st_mtime + 5
        os.utime(parquet_file, (new_mtime, new_mtime))

        fp_after = prepare_sources(config)[0].train_dataset._fingerprint
        assert fp_before != fp_after

    def test_unaffected_by_cluster_column(self, corpus_dir):
        """`cluster_column` must only invalidate the cluster index, never the
        split -- already covered by `TestClusterColumnReuse`, repeated here to
        pin it against this specific fingerprinting mechanism."""
        fp_a = prepare_sources(
            self._config(
                corpus_dir,
                cluster_column="cluster_rep_at_30",
                split_column="cluster_rep_at_30",
                extra_cluster_columns=["cluster_rep_at_30", "cluster_rep_at_50"],
            )
        )[0].train_dataset._fingerprint
        fp_b = prepare_sources(
            self._config(
                corpus_dir,
                cluster_column="cluster_rep_at_50",
                split_column="cluster_rep_at_30",
                extra_cluster_columns=["cluster_rep_at_30", "cluster_rep_at_50"],
            )
        )[0].train_dataset._fingerprint
        assert fp_a == fp_b

    def test_layout_only_rerun_still_hits_cluster_index_cache(self, corpus_dir):
        """Validates the plan's central claim: a stable split fingerprint
        makes `load_or_build_cluster_index`'s cache (keyed on it) stable too,
        with no separate change needed there."""
        config_a = self._config(corpus_dir)
        config_a.num_proc = 1
        config_b = self._config(corpus_dir)
        config_b.num_proc = 2
        index_a = prepare_sources(config_a)[0].train_index
        index_b = prepare_sources(config_b)[0].train_index
        assert index_a.cache_dir == index_b.cache_dir

    def test_roundtrip_is_idempotent(self, corpus_dir):
        """Same config, two separate `prepare_sources` calls: rows must be
        byte-identical (regression guard on the pinned-fingerprint path)."""
        first = prepare_sources(self._config(corpus_dir))[0]
        second = prepare_sources(self._config(corpus_dir))[0]
        assert list(first.train_dataset["sequence"]) == list(
            second.train_dataset["sequence"]
        )
        assert list(first.val_dataset["sequence"]) == list(
            second.val_dataset["sequence"]
        )


class TestBuildEpochPlan:
    """Cross-source epoch planning against real prepared sources."""

    def test_row_ids_and_lengths_stay_aligned_and_in_range(self, dataset_config):
        prepared = prepare_sources(dataset_config)
        total_rows = sum(len(p.train_dataset) for p in prepared)

        plan = build_epoch_plan(
            prepared, epoch=0, seed=dataset_config.seed, dataset_cfg=dataset_config
        )

        assert isinstance(plan, EpochPlan)
        assert len(plan.row_ids) == len(plan.sequence_lengths)
        assert len(plan.row_ids) > 0
        assert plan.row_ids.min() >= 0
        assert plan.row_ids.max() < total_rows
        assert len(set(plan.row_ids.tolist())) == len(plan.row_ids)  # no duplicates

    def test_curriculum_shrinks_epoch_plan(self, dataset_config):
        prepared = prepare_sources(dataset_config)
        early = len(
            build_epoch_plan(prepared, 0, dataset_config.seed, dataset_config).row_ids
        )
        late = len(
            build_epoch_plan(prepared, 5, dataset_config.seed, dataset_config).row_ids
        )
        assert 0 < late < early

    def test_second_source_row_ids_are_offset_past_first_source(self, dataset_config):
        """Regression guard for cross-source global-id offsetting.

        Row ids index the concatenated dataset, so the second source's ids
        must be shifted by the first source's length -- otherwise every
        source silently trains on source 0's rows.
        """
        prepared = prepare_sources(dataset_config)
        # val split uses a fixed, low threshold so every source contributes rows.
        plan = build_epoch_plan(
            prepared,
            epoch=0,
            seed=dataset_config.seed,
            dataset_cfg=dataset_config,
            split="val",
        )
        first_source_len = len(prepared[0].val_dataset)
        assert plan.row_ids.min() >= 0
        assert plan.row_ids.max() < first_source_len + len(prepared[1].val_dataset)
        # With two equally-sized sources contributing, ids should span both ranges.
        assert (plan.row_ids < first_source_len).any()
        assert (plan.row_ids >= first_source_len).any()

    def test_val_plan_size_is_invariant_to_epoch(self, dataset_config):
        """Val's *threshold* ignores epoch (unlike train's curriculum), so the
        number of eligible clusters must stay constant across epochs -- even
        though the exact representative row chosen per cluster still varies
        by epoch, since ``epoch`` also feeds the per-source RNG seed.
        """
        prepared = prepare_sources(dataset_config)
        sizes = {
            epoch: len(
                build_epoch_plan(
                    prepared,
                    epoch=epoch,
                    seed=dataset_config.seed,
                    dataset_cfg=dataset_config,
                    split="val",
                ).row_ids
            )
            for epoch in (0, 3)
        }
        assert sizes[0] == sizes[3]


class TestEndToEnd:
    """Runs the real Trainer over the real data path for a few steps."""

    @pytest.fixture
    def pieces(self, dataset_config, tmp_path):
        from accelerate import Accelerator

        from modules.pretrain.src.config import PretrainConfig
        from modules.pretrain.src.model.modeling_amplify import AMPLIFYModelConfig
        from modules.pretrain.src.optimizer import OptimizerConfig
        from modules.pretrain.src.scheduler import SchedulerConfig
        from modules.pretrain.src.trainer.trainer import DDPConfig, TrainerConfig

        config = PretrainConfig(
            model=AMPLIFYModelConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=128,
                vocab_size=len(VOCAB),
                pad_token_id=VOCAB.index("<pad>"),
                bos_token_id=VOCAB.index("<bos>"),
                eos_token_id=VOCAB.index("<eos>"),
            ),
            tokenizer=_tokenizer_config(),
            optimizer=OptimizerConfig(
                type="AdamW",
                lr=1e-3,
                betas=[0.9, 0.95],
                eps=1e-8,
                weight_decay=0.01,
                fused=False,
            ),
            scheduler=SchedulerConfig(
                type="cosine_with_min_lr",
                num_warmup_steps=1,
                num_training_steps=20,
                kwargs={"min_lr_rate": 0.1},
            ),
            collator=_collator_config(),
            dataset=dataset_config,
            ddp=DDPConfig(),
            dataloader=DataLoaderConfig(
                max_tokens=512,
                dataloader_num_workers=0,
                persistent_workers=False,
                pin_memory=False,
            ),
            trainer=TrainerConfig(
                num_epochs=2,
                gradient_accumulation_steps=2,
                output_dir=tmp_path / "run",
                logging_steps=2,
                eval_steps=50,
                save_steps=50,
                hf_save_steps=50,
                tf32=False,
            ),
        )
        accelerator = Accelerator(
            cpu=True,
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        )
        return config, accelerator

    def test_two_epochs_train_and_checkpoint(self, pieces):
        from modules.pretrain.src.metric.logger import TrainingLogger
        from modules.pretrain.src.model.modeling_amplify import get_amplify_masked_lm
        from modules.pretrain.src.model.tokenizer import get_tokenizer
        from modules.pretrain.src.optimizer import get_optimizer
        from modules.pretrain.src.scheduler import get_scheduler
        from modules.pretrain.src.trainer import Trainer

        config, accelerator = pieces

        model = get_amplify_masked_lm(config.model)
        tokenizer = get_tokenizer(config.tokenizer)
        optimizer = get_optimizer(model=model, config=config.optimizer)
        scheduler = get_scheduler(
            optimizer=optimizer, scheduler_config=config.scheduler
        )
        collator = get_collator(tokenizer=tokenizer, collator_config=config.collator)
        prepared = prepare_sources(config.dataset)
        train_concat = _concat(prepared, "train")

        val_dataloader = build_split_dataloader(
            prepared,
            _concat(prepared, "val"),
            dataset_config=config.dataset,
            collator=collator,
            dataloader_config=config.dataloader,
            split="val",
        )

        trainer = Trainer(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader_provider=lambda epoch: build_split_dataloader(
                prepared,
                train_concat,
                dataset_config=config.dataset,
                collator=collator,
                dataloader_config=config.dataloader,
                split="train",
                epoch=epoch,
            ),
            config=config.trainer,
            val_dataloader=val_dataloader,
            metric_logger=TrainingLogger(),
            tokenizer=tokenizer,
        )

        trainer.train()

        assert trainer.global_step > 0
        assert trainer.current_epoch == 1

        output_dir = config.trainer.output_dir
        checkpoints = sorted(output_dir.glob("checkpoint_epoch_*_step_*"))
        assert checkpoints, "no resumable checkpoint was written"

        records = [
            json.loads(line)
            for line in (output_dir / "metrics.jsonl").read_text().splitlines()
        ]
        assert records, "no metrics were logged"

        losses = [r["train/loss"] for r in records if r.get("train/loss", 0.0) > 0.0]
        assert losses, "every logged train loss was zero"
        assert all(np.isfinite(losses)), "non-finite training loss"
        assert any("eval/val/loss" in r for r in records), "validation never ran"

    def test_dataloader_registration_does_not_leak(self, pieces):
        """Accelerate must not accumulate one train dataloader per epoch.

        ``accelerator.prepare`` appends to ``_dataloaders`` and never removes,
        so before the fix this list grew by one every epoch. That leaks worker
        processes under ``persistent_workers=True`` and shifts the *positional*
        sampler indices that ``save_state``/``load_state`` rely on.
        """
        from modules.pretrain.src.metric.logger import TrainingLogger
        from modules.pretrain.src.model.modeling_amplify import get_amplify_masked_lm
        from modules.pretrain.src.model.tokenizer import get_tokenizer
        from modules.pretrain.src.optimizer import get_optimizer
        from modules.pretrain.src.scheduler import get_scheduler
        from modules.pretrain.src.trainer import Trainer

        config, accelerator = pieces
        config = config.model_copy(
            update={"trainer": config.trainer.model_copy(update={"num_epochs": 4})}
        )

        model = get_amplify_masked_lm(config.model)
        tokenizer = get_tokenizer(config.tokenizer)
        optimizer = get_optimizer(model=model, config=config.optimizer)
        scheduler = get_scheduler(
            optimizer=optimizer, scheduler_config=config.scheduler
        )
        collator = get_collator(tokenizer=tokenizer, collator_config=config.collator)
        prepared = prepare_sources(config.dataset)
        train_concat = _concat(prepared, "train")

        val_dataloader = accelerator.prepare(
            build_split_dataloader(
                prepared,
                _concat(prepared, "val"),
                dataset_config=config.dataset,
                collator=collator,
                dataloader_config=config.dataloader,
                split="val",
            )
        )
        baseline = len(accelerator._dataloaders)

        counts = []

        def provider(epoch):
            dataloader = accelerator.prepare(
                build_split_dataloader(
                    prepared,
                    train_concat,
                    dataset_config=config.dataset,
                    collator=collator,
                    dataloader_config=config.dataloader,
                    split="train",
                    epoch=epoch,
                )
            )
            counts.append(len(accelerator._dataloaders))
            return dataloader

        trainer = Trainer(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader_provider=provider,
            config=config.trainer,
            val_dataloader=val_dataloader,
            metric_logger=TrainingLogger(),
            tokenizer=tokenizer,
        )
        trainer.train()

        assert len(counts) == 4, "expected one dataloader build per epoch"
        # `counts` is sampled inside the provider, i.e. after the new loader is
        # registered but before the stale one is dropped, so a transient overlap
        # of exactly one is expected. The invariant is that it stays BOUNDED:
        # with the fix [2, 3, 3, 3]; without it [2, 3, 4, 5].
        assert max(counts) <= baseline + 2, (
            f"dataloader registrations grew per epoch: {counts} (baseline {baseline})"
        )
        assert counts[1:] == counts[1:2] * (len(counts) - 1), (
            f"registration count must plateau, got {counts}"
        )
        assert len(accelerator._dataloaders) == baseline + 1, (
            "exactly one train dataloader should remain registered after training"
        )

    def test_hf_checkpoint_reloads_without_nan(self, pieces, tmp_path):
        """A saved HF checkpoint must reload with usable RoPE tables.

        RoPE tables are non-persistent buffers, so ``from_pretrained``
        reallocates them with ``torch.empty_like``; they are only correct
        because ``_init_weights`` rebuilds them. The allocator is poisoned
        below because a same-process reload otherwise tends to get the
        just-freed (correct) memory back and hide the bug.
        """
        from modules.pretrain.src.metric.logger import TrainingLogger

        # Import the submodule, not the package: tests/pretrain/test_amplify_model.py
        # installs a bare stub at sys.modules['modules.pretrain.src.model'] during
        # collection, which has no attributes and breaks a package-level import.
        from modules.pretrain.src.model.modeling_amplify import (
            AMPLIFYForMaskedLM,
            get_amplify_masked_lm,
        )
        from modules.pretrain.src.model.tokenizer import get_tokenizer
        from modules.pretrain.src.optimizer import get_optimizer
        from modules.pretrain.src.scheduler import get_scheduler
        from modules.pretrain.src.trainer import Trainer

        config, accelerator = pieces
        model = get_amplify_masked_lm(config.model)
        tokenizer = get_tokenizer(config.tokenizer)
        optimizer = get_optimizer(model=model, config=config.optimizer)
        scheduler = get_scheduler(
            optimizer=optimizer, scheduler_config=config.scheduler
        )

        trainer = Trainer(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dataloader_provider=lambda epoch: [],
            config=config.trainer,
            metric_logger=TrainingLogger(),
            tokenizer=tokenizer,
        )
        path = tmp_path / "hf"
        trainer.save_hf_checkpoint(path)

        poison = [torch.full((4096,), float("nan")) for _ in range(200)]
        del poison

        reloaded = AMPLIFYForMaskedLM.from_pretrained(path)
        rope_buffers = [b for n, b in reloaded.named_buffers() if "rope" in n]
        assert rope_buffers, "expected RoPE buffers on the reloaded model"
        assert not any(torch.isnan(b).any() for b in rope_buffers)

        batch = tokenizer(
            ["MKTAYIAKQRQ", "MGSSHHHHHH"], return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = reloaded(**batch).logits
        assert torch.isfinite(logits).all()


class TestWorkerSeedWiring:
    """Guards that the seed generator is actually attached to the DataLoader.

    ``build_worker_seed_generator`` itself is unit-tested in
    ``test_dataloader.py``; what matters here is that ``build_split_dataloader``
    passes it through. Left as None, PyTorch draws the worker base seed from
    the *global* torch RNG at iterator-creation time, so MLM masks and
    truncation windows depend on how much RNG the main process happened to
    consume rather than on the configured seed. That is what made resume
    non-bit-reproducible: ``load_state`` restores the global RNG faithfully,
    but the rebuilt loader then draws its base seed from a different point in
    that stream.

    ``_base_seed`` is asserted on directly because it is the exact value
    workers derive their NumPy seeds from; it is drawn even at
    ``num_workers=0``, so these tests need no subprocesses.
    """

    @staticmethod
    def _base_seed(loader):
        return iter(loader)._base_seed

    def _loader(self, dataset_config, prepared, epoch=0, split="train"):
        from modules.pretrain.src.model.tokenizer import get_tokenizer

        collator = get_collator(
            tokenizer=get_tokenizer(_tokenizer_config()),
            collator_config=_collator_config(),
        )
        return build_split_dataloader(
            prepared,
            _concat(prepared, split),
            dataset_config=dataset_config,
            collator=collator,
            dataloader_config=DataLoaderConfig(
                max_tokens=512, dataloader_num_workers=0, pin_memory=False
            ),
            split=split,
            epoch=epoch,
        )

    def test_generator_is_attached(self, dataset_config):
        loader = self._loader(dataset_config, prepare_sources(dataset_config))
        assert loader.generator is not None, (
            "DataLoader built without an explicit generator; worker seeds would "
            "fall back to the ambient global torch RNG"
        )

    def test_base_seed_is_invariant_to_ambient_rng(self, dataset_config):
        prepared = prepare_sources(dataset_config)

        torch.manual_seed(0)
        torch.randn(100)
        checkpointed = torch.get_rng_state()
        torch.randn(50)  # the rest of the epoch advances the global stream
        baseline = self._base_seed(self._loader(dataset_config, prepared, epoch=1))

        torch.set_rng_state(checkpointed)  # exactly what load_state does
        resumed = self._base_seed(self._loader(dataset_config, prepared, epoch=1))

        assert baseline == resumed, (
            "worker base seed changed after restoring RNG state and rebuilding "
            "the loader; masking would differ on resume"
        )

    def test_ambient_rng_would_otherwise_leak_in(self, dataset_config):
        """Non-vacuity: without a generator the base seed *does* move."""
        from torch.utils.data import DataLoader

        prepared = prepare_sources(dataset_config)
        loader = self._loader(dataset_config, prepared)
        ungenerated = DataLoader(
            loader.dataset,
            batch_sampler=loader.batch_sampler,
            collate_fn=loader.collate_fn,
        )

        torch.manual_seed(0)
        torch.randn(100)
        checkpointed = torch.get_rng_state()
        torch.randn(50)
        baseline = self._base_seed(ungenerated)
        torch.set_rng_state(checkpointed)
        resumed = self._base_seed(ungenerated)

        assert baseline != resumed, (
            "expected the unfixed path to be RNG-position sensitive; if this "
            "fails the invariance test above proves nothing"
        )


def test_model_package_reexports_resolve():
    """``modules.pretrain.src.model`` is the public import surface.

    Nothing in-tree imports it (evaluate loads checkpoints via
    ``trust_remote_code``), so a rename in ``modeling_amplify.py`` would break
    it silently. Run in a subprocess because ``test_amplify_model.py`` installs
    a bare stub at ``sys.modules['modules.pretrain.src.model']`` during
    collection, which would make a plain import here order-dependent.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import modules.pretrain.src.model as m;"
            "missing = [n for n in m.__all__ if not hasattr(m, n)];"
            "assert not missing, missing",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
