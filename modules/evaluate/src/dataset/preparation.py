"""Prepare development and independent test inputs without owning predictors."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import PreTrainedTokenizerBase

from modules.evaluate.src.artifacts.cache import (
    EmbeddingCache,
    embedding_cache_key,
    weights_fingerprint,
)
from modules.evaluate.src.artifacts.workspace import EvaluationWorkspace, TaskType
from modules.evaluate.src.config.evaluation import ExternalTestSetConfig, PrepareConfig
from modules.evaluate.src.config.shared import (
    BaselineName,
    EmbeddingCacheConfig,
    dataset_source_identity,
)
from modules.evaluate.src.dataset.baseline_data import (
    baseline_context,
    external_baseline_context,
    preprocessing_identity,
)
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    build_dataloaders,
    metric_groups_by_id,
)
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingBackend,
    build_embedding_dataloaders,
)
from modules.evaluate.src.dataset.loading import (
    resolve_evaluation_data,
    resolve_external_data,
)
from modules.evaluate.src.dataset.tokenization import tokenize_dataset
from modules.evaluate.src.dataset.types import (
    CachedFeatures,
    LayerwiseInputs,
    TestData,
    TrainingData,
)
from modules.evaluate.src.model.embeddings import validate_embedding_cache_compat
from modules.evaluate.src.model.layerwise import resolve_layers
from modules.evaluate.src.tasks.registry import get_flat_task

if TYPE_CHECKING:
    from modules.evaluate.src.config.evaluation import LayerwiseConfig
logger = get_logger(__name__)


def _token_positions_by_id(
    dataset: Dataset, config: PrepareConfig
) -> dict[str, list[int]]:
    columns = [config.id_column, config.label_column, "original_positions"]
    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Tokenized test data is missing provenance columns required for residue alignment: {sorted(missing)}."
        )
    positions: dict[str, list[int]] = {}
    for row in dataset.select_columns(columns):
        key = json.dumps(row[config.id_column], allow_nan=False)
        if key in positions:
            raise ValueError(
                f"Token comparisons require unique source protein IDs; duplicate ID {row[config.id_column]!r}."
            )
        positions[key] = [
            position
            for position, label in zip(
                row["original_positions"], row[config.label_column], strict=True
            )
            if position >= 0 and label != -100
        ]
    return positions


def prepare_training_data(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    accelerator: Accelerator,
    seed: int = 710019,
    layerwise_config: LayerwiseConfig | None = None,
    dataset: DatasetDict | None = None,
    baseline_names: list[BaselineName] | None = None,
    group_column: str | None = None,
    validation_group_column: str | None = None,
) -> tuple[TrainingData, TestData]:
    """Prepare development and first-test inputs, optionally extracting frozen features."""
    if config.packed and accelerator.device.type != "cuda":
        raise ValueError("Packed evaluation requires CUDA.")
    if config.max_tokens_per_batch is not None and accelerator.num_processes > 1:
        raise ValueError("max_tokens_per_batch supports a single process only.")
    task = get_flat_task(workspace.task_type)
    logger.info("Preparing training and test data...")
    dataset_id = workspace.dataset_repo_id or workspace.dataset_name
    logger.info("Resolving dataset '%s'...", dataset_id)
    dataset = resolve_evaluation_data(
        dataset_id,
        seed=seed,
        validation_fraction=config.validation_fraction,
        split=config.split,
        dataset=dataset,
        config_name=workspace.config.dataset_config_name,
        revision=workspace.config.dataset_revision,
    )
    shared_context = baseline_context(
        workspace.config,
        dataset,
        baseline_names or [],
        sequence_column=config.sequence_column,
        label_column=config.label_column,
        id_column=config.id_column,
        split_roles=config.split.model_dump(),
        max_length=config.max_length,
        preprocessing=preprocessing_identity(
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            id_column=config.id_column,
            split_roles=config.split.model_dump(),
            max_length=config.max_length,
            random_truncate_by_split=config.random_truncate_by_split,
            seed=seed,
        ),
    )
    num_labels = task.resolve_num_labels(workspace.num_labels, workspace.dataset_name)
    task.validate_num_labels(
        dataset, config.label_column, num_labels, workspace.dataset_name
    )
    logger.info("Tokenizing dataset...")
    test_metric_groups = metric_groups_by_id(
        dataset["test"], group_column, config.id_column
    )
    validation_metric_groups = (
        metric_groups_by_id(
            dataset["validation"], validation_group_column, config.id_column
        )
        if validation_group_column is not None and "validation" in dataset
        else None
    )
    with accelerator.main_process_first():
        dataset = tokenize_dataset(
            dataset,
            tokenizer=tokenizer,
            task_type=workspace.task_type,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            max_length=config.max_length,
            id_column=config.id_column,
            num_proc=config.tokenize_num_proc,
            random_truncate_by_split=config.random_truncate_by_split,
            seed=seed,
        )
    logger.info("Building collator and dataloaders...")
    collator = _build_collator(config, tokenizer, workspace.task_type)
    dataloader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        max_tokens_per_batch=config.max_tokens_per_batch,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        persistent_workers=config.dataloader_persistent_workers,
        seed=seed,
        length_grouped_sampling=config.length_grouped_sampling,
    )
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
        dataset=dataset, collator=collator, config=dataloader_config
    )
    refit_token_dataloaders = {}
    for split in config.split.refit_additional_splits:
        if split not in dataset:
            raise ValueError(
                f"Configured final-refit source split '{split}' was not loaded. Available splits: {sorted(dataset.keys())}."
            )
        split_dataset = DatasetDict({"train": dataset[split]})
        refit_loader, _, _ = build_dataloaders(
            dataset=split_dataset, collator=collator, config=dataloader_config
        )
        if refit_loader is None:
            raise ValueError(f"Configured final-refit source split '{split}' is empty.")
        refit_token_dataloaders[split] = accelerator.prepare(refit_loader)
    if config.packed:
        parameters = inspect.signature(model.forward).parameters
        required = {"cu_seqlens", "max_seqlen"}
        if workspace.task_type in ("sequence_classification", "sequence_regression"):
            required.add("num_sequences")
        if not required <= parameters.keys():
            raise ValueError(
                "Packed evaluation requires an AMPLIFY checkpoint re-uploaded with the current modeling_amplify.py."
            )
    active_layerwise_config = (
        layerwise_config
        if layerwise_config is not None and layerwise_config.enabled
        else None
    )
    if active_layerwise_config is not None:
        validate_embedding_cache_compat(
            task.is_ragged, tokenizer, active_layerwise_config.pooling
        )
    raw_test_dataloader = test_dataloader
    train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader, test_dataloader
    )
    token_dataloaders = (
        {
            "train": train_dataloader,
            "validation": val_dataloader,
            "test": test_dataloader,
        }
        if active_layerwise_config is not None
        else None
    )
    if config.embedding_cache.enabled:
        train_dataloader, val_dataloader, test_dataloader, embedding_extras = (
            prepare_feature_inputs(
                config,
                workspace,
                model,
                tokenizer,
                train_dataloader,
                val_dataloader,
                test_dataloader,
                refit_token_dataloaders,
                raw_test_dataloader,
                accelerator,
                seed,
                cache_config=_layerwise_cache_config(
                    config, active_layerwise_config, model
                ),
            )
        )
    else:
        embedding_extras = {}
    return (
        TrainingData(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            layerwise=LayerwiseInputs(token_dataloaders=token_dataloaders)
            if token_dataloaders is not None
            else None,
            refit_token_dataloaders=refit_token_dataloaders,
            refit_random_truncate_by_split={
                split: config.random_truncate_by_split.get(split, False)
                for split in refit_token_dataloaders
            },
            split=config.split.name,
            validation_metric_groups=validation_metric_groups,
            **embedding_extras,
        ),
        TestData(
            test_dataloader=test_dataloader,
            layerwise=LayerwiseInputs(token_dataloaders=token_dataloaders)
            if token_dataloaders is not None
            else None,
            split=config.split.name,
            baseline_context=shared_context,
            test_metric_groups=test_metric_groups,
            test_group_column=group_column,
            test_original_positions=_token_positions_by_id(dataset["test"], config)
            if workspace.task_type == "token_classification"
            else {},
            embedding=embedding_extras.get("embedding"),
        ),
    )


def _layerwise_cache_config(
    config: PrepareConfig, layerwise_config: LayerwiseConfig | None, model: nn.Module
) -> EmbeddingCacheConfig | None:
    """Cache settings matching live layerwise probe representations."""
    if layerwise_config is None:
        return None
    return config.embedding_cache.model_copy(
        update={
            "layers": resolve_layers(
                layerwise_config.layers, model.config.num_hidden_layers + 1
            ),
            "pooling": layerwise_config.pooling,
            "autocast_dtype": layerwise_config.autocast_dtype,
            "normalize_before_pooling": layerwise_config.normalize_before_pooling,
        }
    )


def _test_cache_key(
    config: PrepareConfig,
    cache_config: EmbeddingCacheConfig,
    *,
    model_id: str,
    dataset_id: str,
    token_test_dataloader: Any,
    model_fingerprint: str,
    seed: int,
) -> str:
    return embedding_cache_key(
        model_id=model_id,
        dataset_id=dataset_id,
        split="test",
        max_length=config.max_length,
        random_truncate=config.random_truncate_by_split.get("test", False),
        seed=seed,
        pooling=cache_config.pooling,
        dtype=cache_config.dtype,
        layers=cache_config.resolved_layers,
        model_fingerprint=model_fingerprint,
        dataset_fingerprint=getattr(
            token_test_dataloader.dataset, "_fingerprint", None
        ),
        autocast_dtype=cache_config.autocast_dtype,
        normalize_before_pooling=cache_config.normalize_before_pooling,
        packed=config.packed,
        length_grouped_sampling=config.length_grouped_sampling,
    )


def prepare_test_data(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    trunk: nn.Module | None,
    tokenizer: PreTrainedTokenizerBase,
    test: TestData,
    external: ExternalTestSetConfig,
    *,
    accelerator: Accelerator,
    seed: int,
    layerwise_config: LayerwiseConfig | None = None,
    dataset: DatasetDict | None = None,
    group_column: str | None = None,
) -> TestData:
    """Prepare an independent external test view without changing development data.

    The fitted model, heads, and tokenizer are reused unchanged. Cached
    runs embed the external set once with the frozen trunk and the same
    pooling/layer settings, under a cache key naming the external dataset.
    """
    logger.info(
        "Preparing external test set '%s' (%s:%s)...",
        external.name,
        external.dataset_repo_id,
        external.split,
    )
    if dataset is None:
        dataset = resolve_external_data(
            external.dataset_repo_id,
            external.split,
            config_name=external.dataset_config_name,
            revision=external.dataset_revision,
        )
    shared_context = (
        external_baseline_context(
            test.baseline_context,
            workspace.config,
            dataset,
            dataset_id=external.dataset_repo_id,
            split=external.split,
            config_name=external.dataset_config_name,
            revision=external.dataset_revision,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            id_column=config.id_column,
            max_length=config.max_length,
            preprocessing=preprocessing_identity(
                sequence_column=config.sequence_column,
                label_column=config.label_column,
                id_column=config.id_column,
                split_roles={"name": external.name, "test": external.split},
                max_length=config.max_length,
                random_truncate_by_split=config.random_truncate_by_split,
                seed=seed,
            ),
        )
        if test.baseline_context
        else None
    )
    task = get_flat_task(workspace.task_type)
    num_labels = task.resolve_num_labels(workspace.num_labels, workspace.dataset_name)
    task.validate_num_labels(dataset, config.label_column, num_labels, external.name)
    test_metric_groups = metric_groups_by_id(
        dataset["test"], group_column, config.id_column
    )
    with accelerator.main_process_first():
        dataset = tokenize_dataset(
            dataset,
            tokenizer=tokenizer,
            task_type=workspace.task_type,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            max_length=config.max_length,
            id_column=config.id_column,
            num_proc=config.tokenize_num_proc,
            random_truncate_by_split=config.random_truncate_by_split,
            seed=seed,
        )
    _, _, raw_test_dataloader = build_dataloaders(
        dataset=dataset,
        collator=_build_collator(config, tokenizer, workspace.task_type),
        config=DataLoaderConfig(
            batch_size=config.batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            persistent_workers=config.dataloader_persistent_workers,
            seed=seed,
            length_grouped_sampling=config.length_grouped_sampling,
        ),
    )
    if raw_test_dataloader is None:
        raise ValueError(
            f"External test set '{external.name}' did not produce a test dataloader."
        )
    token_test_dataloader = accelerator.prepare(raw_test_dataloader)
    embedding = test.embedding
    backend = embedding.backend if embedding is not None else None
    layerwise = (
        replace(
            test.layerwise,
            token_dataloaders={
                **test.layerwise.token_dataloaders,
                "test": token_test_dataloader,
            },
        )
        if test.layerwise is not None
        else None
    )
    if backend is None:
        return replace(
            test,
            test_dataloader=token_test_dataloader,
            layerwise=layerwise,
            baseline_context=shared_context,
            test_metric_groups=test_metric_groups,
            test_group_column=group_column,
            test_original_positions=_token_positions_by_id(dataset["test"], config)
            if workspace.task_type == "token_classification"
            else {},
        )
    if backend.cache is None or backend.model_id is None:
        raise ValueError(
            "Embedding-cache external evaluation requires the cache and model ID recorded by the original Prepare step."
        )
    if trunk is None:
        raise ValueError("Cached test preparation requires the retained frozen trunk.")
    cache_config = (
        _layerwise_cache_config(
            config,
            layerwise_config if layerwise_config and layerwise_config.enabled else None,
            trunk,
        )
        or config.embedding_cache
    )
    model_fingerprint = weights_fingerprint(trunk)
    external_id = dataset_source_identity(
        external.dataset_repo_id,
        external.dataset_config_name,
        external.dataset_revision,
    )
    embedding_dataloader = build_embedding_dataloaders(
        model=trunk,
        splits={
            "test": (
                token_test_dataloader,
                config.random_truncate_by_split.get("test", False),
            )
        },
        accelerator=accelerator,
        cache=backend.cache,
        cache_config=cache_config,
        model_id=backend.model_id,
        dataset_id=external_id,
        max_length=config.max_length,
        seed=seed,
        batch_size=config.batch_size,
        extraction_batch_size=None if config.packed else config.embedding_batch_size,
        embedding_size=backend.hidden_size,
        label_dtype=task.label_dtype,
        is_ragged=task.is_ragged,
        model_fingerprint=model_fingerprint,
        length_grouped_sampling=config.length_grouped_sampling,
    )["test"]
    external_backend = replace(
        backend,
        token_test_dataloader=raw_test_dataloader,
        dataset_id=external_id,
        test_cache_key=_test_cache_key(
            config,
            cache_config,
            model_id=backend.model_id,
            dataset_id=external_id,
            token_test_dataloader=raw_test_dataloader,
            model_fingerprint=model_fingerprint,
            seed=seed,
        ),
        test_dataset_fingerprint=getattr(
            raw_test_dataloader.dataset, "_fingerprint", None
        ),
    )
    return replace(
        test,
        test_dataloader=embedding_dataloader,
        embedding=CachedFeatures(backend=external_backend),
        layerwise=layerwise,
        baseline_context=shared_context,
        test_metric_groups=test_metric_groups,
        test_group_column=group_column,
        test_original_positions=_token_positions_by_id(dataset["test"], config)
        if workspace.task_type == "token_classification"
        else {},
    )


def prepare_feature_inputs(
    config: PrepareConfig,
    workspace: EvaluationWorkspace,
    model: nn.Module,
    tokenizer: Any,
    train_dataloader: Any,
    val_dataloader: Any,
    test_dataloader: Any,
    refit_token_dataloaders: dict[str, Any],
    raw_test_dataloader: Any,
    accelerator: Accelerator,
    seed: int,
    cache_config: EmbeddingCacheConfig | None = None,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Extract and cache frozen features without constructing a trainable head."""
    cache_config = cache_config or config.embedding_cache
    task = get_flat_task(workspace.task_type)
    is_ragged = task.is_ragged
    validate_embedding_cache_compat(is_ragged, tokenizer, cache_config.pooling)
    hidden_size = model.config.hidden_size
    layers = cache_config.resolved_layers
    embedding_size = hidden_size * len(layers)
    model_id = workspace.model_repo_id or workspace.model_name
    dataset_id = dataset_source_identity(
        workspace.dataset_repo_id or workspace.dataset_name,
        workspace.config.dataset_config_name,
        workspace.config.dataset_revision,
    )
    cache = EmbeddingCache(workspace.embeddings_path, cache_config.max_cache_gb)
    token_test_dataloader = raw_test_dataloader
    model_fingerprint = weights_fingerprint(model)
    test_random_truncate = config.random_truncate_by_split.get("test", False)
    test_cache_key = _test_cache_key(
        config,
        cache_config,
        model_id=model_id,
        dataset_id=dataset_id,
        token_test_dataloader=token_test_dataloader,
        model_fingerprint=model_fingerprint,
        seed=seed,
    )
    label_dtype = task.label_dtype
    embedding_dataloaders = build_embedding_dataloaders(
        model=model,
        splits={
            "train": (
                train_dataloader,
                config.random_truncate_by_split.get("train", True),
            ),
            "validation": (
                val_dataloader,
                config.random_truncate_by_split.get("validation", False),
            ),
            "test": (
                test_dataloader,
                config.random_truncate_by_split.get("test", False),
            ),
            **{
                split: (dataloader, config.random_truncate_by_split.get(split, False))
                for split, dataloader in refit_token_dataloaders.items()
            },
        },
        accelerator=accelerator,
        cache=cache,
        cache_config=cache_config,
        model_id=model_id,
        dataset_id=dataset_id,
        max_length=config.max_length,
        seed=seed,
        batch_size=config.batch_size,
        extraction_batch_size=None if config.packed else config.embedding_batch_size,
        embedding_size=embedding_size,
        label_dtype=label_dtype,
        is_ragged=is_ragged,
        model_fingerprint=model_fingerprint,
        length_grouped_sampling=config.length_grouped_sampling,
    )
    refit_embedding_dataloaders = {}
    for split in refit_token_dataloaders:
        dataloader = embedding_dataloaders[split]
        if dataloader is None:
            raise ValueError(
                f"Could not build embeddings for final-refit split '{split}'."
            )
        refit_embedding_dataloaders[split] = dataloader
    embedding_extras = {
        "embedding": CachedFeatures(
            backend=EmbeddingBackend(
                token_test_dataloader=token_test_dataloader,
                token_train_dataloader=train_dataloader,
                token_val_dataloader=val_dataloader,
                train_random_truncate=config.random_truncate_by_split.get(
                    "train", True
                ),
                validation_random_truncate=config.random_truncate_by_split.get(
                    "validation", False
                ),
                extraction_batch_size=None
                if config.packed
                else config.embedding_batch_size,
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
                max_length=config.max_length,
                test_random_truncate=test_random_truncate,
                length_grouped_sampling=config.length_grouped_sampling,
                test_dataset_fingerprint=getattr(
                    token_test_dataloader.dataset, "_fingerprint", None
                ),
            )
        ),
        "refit_dataloaders": refit_embedding_dataloaders,
    }
    return (
        embedding_dataloaders["train"],
        embedding_dataloaders["validation"],
        embedding_dataloaders["test"],
        embedding_extras,
    )


def _build_collator(
    config: PrepareConfig, tokenizer: PreTrainedTokenizerBase, task_type: TaskType
) -> EvaluationCollator:
    return EvaluationCollator(
        tokenizer=tokenizer,
        task_type=task_type,
        config=EvaluationCollatorConfig(
            pad_to_multiple_of=config.pad_to_multiple_of,
            packed=config.packed,
            label_column=config.label_column,
            id_column=config.id_column,
        ),
    )
