import inspect
import itertools
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
from datasets import Dataset as HFDataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from modules.data.src.dataset.dataset import Dataset
from modules.data.src.utils.io_utils import override_or_default, resolve_path

logger = logging.getLogger(__name__)

PARQUET_OPTIONS = {
    "compression": "zstd",
    "write_statistics": False,
    "use_dictionary": False,
}

# Maps ScoreConfig.mixed_precision values to the corresponding torch dtype.
MIXED_PRECISION_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


class ScoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    override_source: Path | None = None
    output_base_path: Path | None = None
    inference_model: str = "flair-bio/amplify-350m"
    sequence_name_col: str = "sequence_id"
    sequence_column: str = "sequence"
    score_column: str = "RED"
    packed: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    max_length: int = 2048
    pad_to_multiple_of: int = 8
    dataloader_workers: int = 4
    batch_size: int = 16

    # Skip already scored shards to allow easy resumption.
    resume: bool = True

    # SLURM / Sharding Overrides
    shard_index: int | None = None
    shard_count: int | None = None


class _ScoreCollator:
    """Tokenizes a batch in padded or packed form."""

    def __init__(
        self,
        name_column: str,
        sequence_column: str,
        tokenizer: Any,
        packed: bool,
        max_length: int | None,
        pad_to_multiple_of: int,
        shard_name: str,
    ) -> None:
        self.name_column = name_column
        self.sequence_column = sequence_column
        self.tokenizer = tokenizer
        self.packed = packed
        self.max_length = max_length
        self.pad_to_multiple_of = max(1, pad_to_multiple_of)
        self.shard_name = shard_name

        if hasattr(self.tokenizer, "model_max_length"):
            self.tokenizer.model_max_length = int(1e8)

    def _pack(self, sequences: list[str]) -> dict[str, Any]:
        features = self.tokenizer(
            sequences,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            remove_ambiguous=True,
        )
        input_ids_list = features["input_ids"]
        seqlens = [len(seq) for seq in input_ids_list]

        def flatten(chunks: Any, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(
                list(itertools.chain.from_iterable(chunks)), dtype=dtype
            ).unsqueeze(0)

        return {
            "input_ids": flatten(input_ids_list, torch.long),
            "attention_mask": None,
            "cu_seqlens": torch.tensor(
                [0, *itertools.accumulate(seqlens)], dtype=torch.int32
            ),
            "position_ids": flatten((range(n) for n in seqlens), torch.long),
            "max_seqlen": max(seqlens) if seqlens else 0,
        }

    def _pad(self, sequences: list[str]) -> dict[str, Any]:
        features = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
            remove_ambiguous=True,
        )
        return {
            "input_ids": features["input_ids"],
            "attention_mask": features.get("attention_mask"),
            "cu_seqlens": None,
            "position_ids": features.get("position_ids"),
            "max_seqlen": None,
        }

    def __call__(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        sequences = [item[self.sequence_column] for item in inputs]
        names = [item[self.name_column] for item in inputs]
        batch = self._pack(sequences) if self.packed else self._pad(sequences)
        return {**batch, self.name_column: names}


class ScoreStep:
    def __init__(self, config: ScoreConfig) -> None:
        self.config = config

    @staticmethod
    def _balance_shards(shards: list[Path], count: int) -> list[list[Path]]:
        """Split shards into buckets balanced by row count."""
        if count <= 0:
            return []
        sized = sorted(
            ((pq.ParquetFile(s).metadata.num_rows, s) for s in shards),
            key=lambda item: item[0],
            reverse=True,
        )
        buckets: list[list[Path]] = [[] for _ in range(count)]
        loads = [0] * count
        for rows, path in sized:
            i = loads.index(min(loads))
            buckets[i].append(path)
            loads[i] += rows
        return buckets

    def _score_output_path(self, shard_path: Path, output_dir: Path) -> Path:
        return output_dir / f"{shard_path.stem}.parquet"

    def _get_assigned_shards(
        self, all_shards: list[Path], output_dir: Path
    ) -> list[Path]:
        """Assign pending shards to this SLURM array task."""
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", "0"))

        index = (
            self.config.shard_index
            if self.config.shard_index is not None
            else task_id - task_min
        )
        count = (
            self.config.shard_count
            if self.config.shard_count is not None
            else max(1, task_max - task_min + 1)
        )
        if not 0 <= index < count:
            raise ValueError(
                f"shard_index {index} is out of range for shard_count {count}."
            )

        # Remove already-scored shards before partitioning.
        pending = all_shards
        if self.config.resume:
            skipped = [
                s for s in all_shards if self._score_output_path(s, output_dir).exists()
            ]
            for shard in skipped:
                logger.info("Skipping %s, output already exists.", shard.name)
            pending = [s for s in all_shards if s not in skipped]

        assigned = self._balance_shards(pending, count)[index]
        logger.info(
            "Task %d/%d assigned %d of %d pending shards (%d already scored)",
            index + 1,
            count,
            len(assigned),
            len(pending),
            len(all_shards) - len(pending),
        )
        return assigned

    @staticmethod
    def _sweep_stale_tmp(output_dir: Path) -> None:
        """Remove stale .tmp files from killed workers."""
        cutoff = time.time() - 24 * 3600
        for tmp in output_dir.glob("*.parquet.tmp"):
            if tmp.stat().st_mtime < cutoff:
                logger.info("Removing stale temp file %s", tmp.name)
                tmp.unlink(missing_ok=True)

    def _score_batch(
        self,
        batch: dict[str, Any],
        model: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Run one batch, extract embeddings, and calculate RED score."""
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = (
            batch["attention_mask"].to(device, non_blocking=True)
            if batch.get("attention_mask") is not None
            else None
        )
        cu_seqlens = (
            batch["cu_seqlens"].to(device, non_blocking=True)
            if batch.get("cu_seqlens") is not None
            else None
        )
        position_ids = (
            batch["position_ids"].to(device, non_blocking=True)
            if batch.get("position_ids") is not None
            else None
        )

        forward = model.forward if hasattr(model, "forward") else model.__call__
        params = inspect.signature(forward).parameters

        kwargs = {"output_hidden_states": True}
        kwargs["src" if "src" in params else "input_ids"] = input_ids
        if attention_mask is not None:
            kwargs["pad_mask" if "pad_mask" in params else "attention_mask"] = (
                attention_mask
            )
        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        if batch.get("max_seqlen") is not None:
            kwargs["max_seqlen"] = batch["max_seqlen"]

        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in params}

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, dtype=dtype),
        ):
            output = model(**kwargs)

        hidden_states = (
            output["hidden_states"]
            if isinstance(output, dict)
            else output.hidden_states
        )
        hidden_states = hidden_states[-1].float()
        scores = []

        num_samples = len(batch[self.config.sequence_name_col])
        for i in range(num_samples):
            # Extract tokens
            if self.config.packed:
                assert cu_seqlens is not None
                start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
                tokens = hidden_states[0][start:end]
            else:
                tokens = (
                    hidden_states[i][attention_mask[i].bool()]
                    if attention_mask is not None
                    else hidden_states[i]
                )

            # Compute Cosine Similarity (RED Score)
            n_tokens = tokens.shape[0]
            if n_tokens <= 1:
                scores.append(0.0)
                continue

            unit = tokens / (torch.norm(tokens, dim=1, keepdim=True) + 1e-8)
            n_off_diag = n_tokens * (n_tokens - 1)
            unit_sum = unit.sum(dim=0)
            mean_sim = (torch.dot(unit_sum, unit_sum) - n_tokens) / n_off_diag
            scores.append(1.0 - mean_sim.item())

        return {
            self.config.sequence_name_col: batch[self.config.sequence_name_col],
            self.config.score_column: scores,
        }

    def _worker_fn(
        self, local_rank: int, gpu_buckets: list[list[Path]], output_dir: Path
    ) -> None:
        """Worker function executed per GPU."""
        # Set up logging for spawned children.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        rank_shards = gpu_buckets[local_rank]
        if not rank_shards:
            return

        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Determine native PyTorch precision type ("no" falls back to fp32).
        dtype = MIXED_PRECISION_DTYPES.get(self.config.mixed_precision, torch.float32)

        # Load model to device (cache is pre-warmed by parent).
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.inference_model,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            self.config.inference_model, trust_remote_code=True, local_files_only=True
        ).to(device)
        model.eval()

        schema = pa.schema(
            [
                pa.field(self.config.sequence_name_col, pa.string()),
                pa.field(self.config.score_column, pa.float16()),
            ]
        )

        for shard_path in rank_shards:
            final_output = self._score_output_path(shard_path, output_dir)

            if self.config.resume and final_output.exists():
                logger.info(
                    f"Worker {local_rank} skipping {shard_path.name}, output exists."
                )
                continue

            # Use unique temp filenames to prevent collisions.
            job_id = os.environ.get("SLURM_JOB_ID", "local")
            tmp_output = final_output.with_suffix(
                f".{job_id}.{os.getpid()}.parquet.tmp"
            )
            tmp_output.unlink(missing_ok=True)

            shard_dataset = HFDataset.from_parquet(str(shard_path))
            dataloader = DataLoader(
                shard_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.dataloader_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=_ScoreCollator(
                    self.config.sequence_name_col,
                    self.config.sequence_column,
                    tokenizer,
                    self.config.packed,
                    self.config.max_length,
                    self.config.pad_to_multiple_of,
                    shard_path.stem,
                ),
            )

            try:
                with pq.ParquetWriter(
                    tmp_output, schema=schema, **PARQUET_OPTIONS
                ) as writer:
                    pbar = tqdm(
                        dataloader,
                        desc=f"GPU {local_rank} - {shard_path.name}",
                        position=local_rank,
                    )
                    for batch in pbar:
                        columns = self._score_batch(batch, model, device, dtype)
                        writer.write_table(pa.table(columns, schema=schema))

                # Atomic rename for clean recovery.
                tmp_output.rename(final_output)
            except Exception as e:
                tmp_output.unlink(missing_ok=True)
                logger.error(f"Worker {local_rank} failed on {shard_path.name}: {e}")
                raise

    def run(self, dataset: Dataset | None = None) -> None:
        # 1. Resolve source and output directories.
        if dataset is not None:
            dataset.setup_directories()

        source_default = (
            dataset.tmp_path / f"{dataset.name}_parquet_shards" if dataset else None
        )
        output_default = dataset.tmp_path if dataset else None

        if self.config.override_source is None and source_default is None:
            raise ValueError(
                "ScoreConfig requires override_source unless run(dataset) is used."
            )
        if self.config.output_base_path is None and output_default is None:
            raise ValueError(
                "ScoreConfig requires output_base_path unless run(dataset) is used."
            )

        assert source_default is not None or self.config.override_source is not None
        assert output_default is not None or self.config.output_base_path is not None

        source_dir = resolve_path(
            override=self.config.override_source,
            default=source_default or Path(),
            label="Score source directory",
            required=True,
        )
        assert source_dir is not None

        output_base = override_or_default(
            override=self.config.output_base_path, default=output_default or Path()
        )
        output_dir = (
            output_base / f"{dataset.name}_scores"
            if dataset is not None
            else output_base
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.packed and self.config.mixed_precision == "no":
            raise ValueError(
                "Packed scoring requires fp16 or bf16 mixed precision. "
                "Set mixed_precision='bf16' or disable packed mode."
            )

        self._sweep_stale_tmp(output_dir)

        all_shards = sorted(source_dir.glob("*.parquet"))
        if not all_shards:
            raise FileNotFoundError(f"No parquet shards found in {source_dir}")

        # 2. Assign shards deterministically to this SLURM task.
        assigned_shards = self._get_assigned_shards(all_shards, output_dir)
        if not assigned_shards:
            logger.info("No shards assigned to this task or all completed; exiting.")
            return

        # 3. Bucket shards for local multi-GPU execution.
        num_gpus = torch.cuda.device_count() or 1
        gpu_buckets = self._balance_shards(assigned_shards, num_gpus)

        # Warm HF cache to prevent parallel downloads.
        AutoTokenizer.from_pretrained(
            self.config.inference_model, trust_remote_code=True, use_fast=True
        )
        AutoModel.from_pretrained(self.config.inference_model, trust_remote_code=True)

        logger.info(
            f"Spawning {num_gpus} local workers to process {len(assigned_shards)} shards."
        )

        # 4. Launch isolated processes per GPU to prevent cascade failures.
        ctx = mp.get_context("spawn")
        processes = [
            ctx.Process(target=self._worker_fn, args=(rank, gpu_buckets, output_dir))
            for rank in range(num_gpus)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        failed_ranks = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed_ranks:
            raise RuntimeError(
                f"Worker rank(s) {failed_ranks} failed; other ranks' completed "
                "shards were still written and are safe to resume from."
            )
