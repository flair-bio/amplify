"""
Run the evaluation preparation workflow.

The step is ordered to minimize work and keep the decisions explicit:
validate the workspace, load the tokenizer, load and tokenize the dataset once,
derive the task-specific label count, build the collator/DataLoaders, and only
then instantiate the model wrapper that matches the task type.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modules.evaluate.src.dataset.collator import (
    EvaluationCollatorConfig,
    EvaluationCollator,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    DatasetSplitConfig,
    build_dataloaders,
    load_evaluation_dataset,
    tokenize_dataset,
)
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace, TaskType
from modules.evaluate.src.schemas.artifacts import PreparedArtifacts
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.utils.embed import (
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingCacheConfig,
    build_embedding_dataloaders,
    embedding_cache_key,
    validate_embedding_cache_compat,
    weights_fingerprint,
)

# accelerate's logger adapter logs INFO/DEBUG only on the main process by
# default, preventing duplicate lines when running under distributed launch.
logger = get_logger(__name__)


class PrepareConfig(BaseModel):
    """Config. for the prepare step in evaluation."""

    model_config = ConfigDict(extra="forbid")
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    batch_size: int = 16
    packed: bool = False
    max_tokens_per_batch: int | None = Field(None, gt=0)
    embedding_batch_size: int = Field(
        128,
        gt=0,
        description=(
            "Batch size used only while extracting frozen-trunk embeddings. "
            "Increase until the trunk approaches device memory capacity."
        ),
    )
    dataloader_num_workers: int | None = Field(
        None,
        description=(
            "Number of dataloader worker processes. Defaults to "
            "min(os.cpu_count(), num_gpus * 4) on GPU runs and up to 4 "
            "workers on CPU-only runs when not set."
        ),
    )
    max_length: int = Field(
        512, description="Maximum sequence length after truncation."
    )
    pad_to_multiple_of: int | None = 8
    sequence_column: str = "sequence"
    label_column: str = Field(
        "targets",
        description=(
            "Dataset column containing the label(s). The biomap-research "
            "evaluation datasets (e.g. localization_prediction, metal_ion_binding, "
            "ssp_q3) all use 'targets', not 'label'."
        ),
    )
    id_column: str = Field(
        "id",
        description=(
            "Dataset column containing a per-example identifier, passed "
            "through the batch so downstream prediction/scoring steps can key "
            "predictions/labels back to their source example. Falls back "
            "to a positional index when the dataset has no such column."
        ),
    )
    tokenize_num_proc: int | None = Field(
        None,
        description=(
            "Number of processes used to tokenize the dataset up front. "
            "Defaults to min(os.cpu_count(), num_gpus * 4) on GPU runs and "
            "up to 4 workers on CPU-only runs when not set."
        ),
    )
    random_truncate_by_split: dict[str, bool] = Field(
        default_factory=lambda: {
            "train": True,
            "validation": False,
            "test": False,
        },
        description=(
            "Whether sequences longer than max_length are cropped from a "
            "random window, keyed by split. Defaults to random train crops "
            "and deterministic validation/test crops."
        ),
    )
    validation_fraction: float = Field(
        0.1,
        description=(
            "Fraction of 'train' held out to synthesize a 'validation' split "
            "when the dataset doesn't already have one."
        ),
    )
    split: DatasetSplitConfig = Field(
        default_factory=DatasetSplitConfig,
        description=(
            "Named dataset split method and source split names to map onto "
            "the pipeline's train/validation/test roles."
        ),
    )
    length_grouped_sampling: bool = Field(
        False,
        description=(
            "Batch examples of similar tokenized length together (via "
            "LengthGroupedSampler) instead of pure random shuffling. Reduces "
            "the amount of padding the collator has to add per batch, at the "
            "cost of batches no longer being uniformly randomly composed."
        ),
    )
    embedding_cache: EmbeddingCacheConfig = Field(default_factory=EmbeddingCacheConfig)


class PrepareStep:
    def __init__(self, config: PrepareConfig) -> None:
        self.config = config

    def run(
        self,
        workspace: EvaluationWorkspace,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
    ) -> PreparedArtifacts:
        # The pipeline passes one shared instance; keep the fallback for direct
        # step use in tests and small standalone callers.
        if accelerator is None:
            accelerator = Accelerator()
        if self.config.packed and accelerator.device.type != "cuda":
            raise ValueError("Packed evaluation requires CUDA.")
        if (
            self.config.max_tokens_per_batch is not None
            and accelerator.num_processes > 1
        ):
            raise ValueError("max_tokens_per_batch supports a single process only.")

        logger.info("Running Prepare step...")

        logger.info("Setting up workspace directories...")
        workspace.setup_directories()

        model_id = workspace.model_repo_id or workspace.model_name
        dataset_id = workspace.dataset_repo_id or workspace.dataset_name

        # Load the tokenizer
        logger.info("Loading tokenizer for '%s'...", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Load the evaluation dataset from HF hub or local cache
        logger.info("Loading dataset '%s'...", dataset_id)
        dataset = load_evaluation_dataset(
            dataset_id,
            seed=seed,
            validation_fraction=self.config.validation_fraction,
            split=self.config.split,
        )

        task = get_task(workspace.task_type)
        num_labels = task.resolve_num_labels(
            workspace.num_labels, workspace.dataset_name
        )
        # Config-level validation can't see real label values, only the
        # declared contract; check the resolved num_labels against the
        # dataset's actual (pre-tokenization) label values here, before a
        # mismatch surfaces later as a cryptic index-out-of-bounds error.
        task.validate_num_labels(
            dataset, self.config.label_column, num_labels, workspace.dataset_name
        )

        # Tokenize the dataset with the model's tokenizer once
        logger.info("Tokenizing dataset...")
        dataset = tokenize_dataset(
            dataset,
            tokenizer=tokenizer,
            task_type=workspace.task_type,
            sequence_column=self.config.sequence_column,
            label_column=self.config.label_column,
            max_length=self.config.max_length,
            id_column=self.config.id_column,
            num_proc=self.config.tokenize_num_proc,
            random_truncate_by_split=self.config.random_truncate_by_split,
            seed=seed,
        )

        # Build the appropriate collator and dataloaders for the task type
        logger.info("Building collator and dataloaders...")
        collator = self._build_collator(tokenizer, workspace.task_type)
        train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
            dataset=dataset,
            collator=collator,
            config=DataLoaderConfig(
                batch_size=self.config.batch_size,
                max_tokens_per_batch=self.config.max_tokens_per_batch,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                persistent_workers=self.config.dataloader_persistent_workers,
                seed=seed,
                length_grouped_sampling=self.config.length_grouped_sampling,
            ),
        )  # We end up with a dataloader of [tokenized_sequence]-[label] for each split

        logger.info(
            "Loading model '%s' for task '%s'...", model_id, workspace.task_type
        )

        # Get the task-specific model type (e.g. XXXforSequenceClassification) and build the model
        model = task.build_model(model_id, num_labels, workspace.model_kwargs)
        if self.config.packed:
            parameters = inspect.signature(model.forward).parameters
            required = {"cu_seqlens", "max_seqlen"}
            if workspace.task_type in (
                "sequence_classification",
                "sequence_regression",
            ):
                required.add("num_sequences")
            if not required <= parameters.keys():
                raise ValueError(
                    "Packed evaluation requires an AMPLIFY checkpoint re-uploaded "
                    "with the current modeling_amplify.py."
                )
        # Keep the model unwrapped for TuneStep (so requires_grad can still be
        # configured before accelerator.prepare), but place it on the same
        # device that accelerator-prepared dataloaders will yield tensors on.
        model = model.to(accelerator.device)

        # Leave the model raw/unwrapped here: TuneStep needs to set
        # requires_grad on it before wrapping, and accelerator.prepare()
        # must only be called once on the model.
        raw_test_dataloader = test_dataloader  # Necessary to keep a token-based test dataloader for PredictStep controls that need real tokens or a re-embeddable trunk.
        train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
            train_dataloader, val_dataloader, test_dataloader
        )

        # We have two downstream options for tuning, depending on user-set configuration:
        # 1) If we're using the HF-defined model, we need to pass tokenized sequences.
        #    Either of these two cases will be handled by the TuneStep and PredictStep:
        #    - if tuning, the tune step will take tokenized sequences, pass them through the model, and tune the head, then the PredictStep will take tokenized sequences, pass them through the model, and get predictions
        #    - if not tuning, the predict step will just take tokenized sequences, pass them through the model, and get predictions
        # 2) If we're using just the trunk of the model and defining our own probe, we want to get embeddings from the trunk and pass those to the probe.
        #    (the predict step will take embeddings, pass them through the probe, and get predictions)
        if self.config.embedding_cache.enabled:
            # Swaps the token-based model/dataloaders above for pooled-embedding
            # equivalents and adds an embedding_backend (trunk/pooling/etc.,
            # see EmbeddingBackend) that PreparedArtifacts only needs in the
            # embedding-cache case.
            (
                model,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                embedding_extras,
            ) = self._apply_embedding_cache(
                workspace,
                model,
                tokenizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                raw_test_dataloader,
                accelerator,
                seed,
                num_labels,
            )
        else:
            embedding_extras = {}

        # Pass the model, tokenizer, and dataloaders (tokenized or pooled-embedding) to the next step (either TuneStep or PredictStep)
        return PreparedArtifacts(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            split=self.config.split.name,
            **embedding_extras,
        )

    def _apply_embedding_cache(
        self,
        workspace: EvaluationWorkspace,
        model: nn.Module,
        tokenizer: Any,
        train_dataloader: Any,
        val_dataloader: Any,
        test_dataloader: Any,
        raw_test_dataloader: Any,
        accelerator: Accelerator,
        seed: int,
        num_labels: int,
    ) -> tuple[nn.Module, Any, Any, Any, dict[str, Any]]:
        """Replace token dataloaders + full trunk with pooled-embedding
        dataloaders + a cached-embedding probe head, computing (and disk-caching
        for sequence-level splits) each split's frozen-trunk embedding exactly once.

        Only valid for a frozen trunk (mode='tune_probe'): a
        cross-step validator in EvaluationConfig rejects the combination of
        embedding_cache.enabled=True outside tune_probe mode, since a trained
        trunk's output isn't fixed and caching it would silently freeze
        stale features into every subsequent trial.
        """
        cache_config = self.config.embedding_cache
        task = get_task(workspace.task_type)
        is_ragged = task.is_ragged
        validate_embedding_cache_compat(is_ragged, tokenizer, cache_config.pooling)
        hidden_size = model.config.hidden_size
        layers = cache_config.resolved_layers
        embedding_size = hidden_size * len(layers)
        model_id = workspace.model_repo_id or workspace.model_name
        dataset_id = workspace.dataset_repo_id or workspace.dataset_name
        cache = EmbeddingCache(workspace.embeddings_path, cache_config.max_cache_gb)
        # The original token-based test dataloader, kept for PredictStep
        # controls/options that need real tokens or a re-embeddable trunk
        # ("random_trunk", "scrambled_sequences", "store_sequences") that a
        # pooled-embedding-only test_dataloader can no longer provide.
        token_test_dataloader = raw_test_dataloader
        model_fingerprint = weights_fingerprint(model)
        test_random_truncate = self.config.random_truncate_by_split.get("test", False)
        test_cache_key = embedding_cache_key(
            model_id=model_id,
            dataset_id=dataset_id,
            split="test",
            max_length=self.config.max_length,
            random_truncate=test_random_truncate,
            seed=seed,
            pooling=cache_config.pooling,
            dtype=cache_config.dtype,
            layers=layers,
            model_fingerprint=model_fingerprint,
            dataset_fingerprint=getattr(
                token_test_dataloader.dataset, "_fingerprint", None
            ),
            autocast_dtype=cache_config.autocast_dtype,
            normalize_before_pooling=cache_config.normalize_before_pooling,
            packed=self.config.packed,
        )

        label_dtype = (
            torch.float if workspace.task_type == "sequence_regression" else torch.long
        )

        embedding_dataloaders = build_embedding_dataloaders(
            model=model,
            splits={
                "train": (
                    train_dataloader,
                    self.config.random_truncate_by_split.get("train", True),
                ),
                "validation": (
                    val_dataloader,
                    self.config.random_truncate_by_split.get("validation", False),
                ),
                "test": (
                    test_dataloader,
                    self.config.random_truncate_by_split.get("test", False),
                ),
            },
            accelerator=accelerator,
            cache=cache,
            cache_config=cache_config,
            model_id=model_id,
            dataset_id=dataset_id,
            max_length=self.config.max_length,
            seed=seed,
            batch_size=self.config.batch_size,
            extraction_batch_size=(
                None if self.config.packed else self.config.embedding_batch_size
            ),
            embedding_size=embedding_size,
            label_dtype=label_dtype,
            is_ragged=is_ragged,
            model_fingerprint=model_fingerprint,
        )

        probe_model = EmbeddingProbeHead(
            embedding_size, num_labels, workspace.task_type
        )
        probe_model = probe_model.to(accelerator.device)
        embedding_extras = {
            "embedding_backend": EmbeddingBackend(
                trunk=model,
                token_test_dataloader=token_test_dataloader,
                pooling=cache_config.pooling,
                layers=layers,
                dtype=cache_config.dtype,
                hidden_size=embedding_size,
                autocast_dtype=cache_config.autocast_dtype,
                normalize_before_pooling=cache_config.normalize_before_pooling,
                test_cache_key=test_cache_key,
                cache=cache,
                cache_policy=cache_config.cache_policy,
                model_id=model_id,
                dataset_id=dataset_id,
                max_length=self.config.max_length,
                test_random_truncate=test_random_truncate,
                test_dataset_fingerprint=getattr(
                    token_test_dataloader.dataset, "_fingerprint", None
                ),
            )
        }
        return (
            probe_model,
            embedding_dataloaders["train"],
            embedding_dataloaders["validation"],
            embedding_dataloaders["test"],
            embedding_extras,
        )

    def _build_collator(
        self, tokenizer: PreTrainedTokenizerBase, task_type: TaskType
    ) -> EvaluationCollator:
        return EvaluationCollator(
            tokenizer=tokenizer,
            task_type=task_type,
            config=EvaluationCollatorConfig(
                pad_to_multiple_of=self.config.pad_to_multiple_of,
                packed=self.config.packed,
                label_column=self.config.label_column,
                id_column=self.config.id_column,
            ),
        )
