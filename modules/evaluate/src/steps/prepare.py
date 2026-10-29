"""
Run the evaluation preparation workflow.

The step is ordered to minimize work and keep the decisions explicit:
validate the workspace, load the tokenizer, load and tokenize the dataset once,
derive the task-specific label count, build the collator/DataLoaders, and only
then instantiate the model wrapper that matches the task type.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import DatasetDict
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modules.evaluate.src.contracts.artifacts import (
    EmbeddingArtifacts,
    LayerwiseArtifacts,
    PreparedArtifacts,
    PreparedFeatures,
)
from modules.evaluate.src.contracts.config import (
    BaselineName,
    EmbeddingCacheConfig,
    ExternalTestSetConfig,
    PrepareConfig,
    dataset_source_identity,
)
from modules.evaluate.src.dataset.baseline_data import (
    baseline_context,
    external_baseline_context,
)
from modules.evaluate.src.dataset.collator import (
    EvaluationCollator,
    EvaluationCollatorConfig,
)
from modules.evaluate.src.dataset.dataloader import (
    DataLoaderConfig,
    build_dataloaders,
    resolve_evaluation_data,
    resolve_external_data,
    tokenize_dataset,
)
from modules.evaluate.src.dataset.embedding_cache import (
    EmbeddingCache,
    embedding_cache_key,
    load_tensor_cache,
    validate_tensor_cache,
    weights_fingerprint,
)
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingBackend,
    build_embedding_dataloaders,
)
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace, TaskType
from modules.evaluate.src.model.embeddings import validate_embedding_cache_compat
from modules.evaluate.src.model.heads import EmbeddingProbeHead
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel, resolve_layers
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.tasks.base import StructuredTask

if TYPE_CHECKING:
    from modules.evaluate.src.contracts.config import LayerwiseConfig

# accelerate's logger adapter logs INFO/DEBUG only on the main process by
# default, preventing duplicate lines when running under distributed launch.
logger = get_logger(__name__)


class PrepareStep:
    def __init__(self, config: PrepareConfig) -> None:
        self.config = config

    def run(
        self,
        workspace: EvaluationWorkspace,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
        layerwise_config: LayerwiseConfig | None = None,
        dataset: DatasetDict | None = None,
        baseline_names: list[BaselineName] | None = None,
        head_type: str | None = None,
    ) -> PreparedArtifacts:
        """Load task data and models, optionally replacing tokens with features.

        Layerwise mode attaches independent heads to the frozen trunk. When
        caching is enabled, its layer and pooling settings govern extraction,
        and heads are attached afterwards. Original token batches remain
        available for controls that change the input or trunk. A supplied
        raw *dataset* with resolved roles replaces loading and validation-split
        synthesis.
        """
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

        logger.info("Setting up workspace directories...")
        workspace.setup_directories()
        task = get_task(workspace.task_type)
        if isinstance(task, StructuredTask):
            prepared = task.prepare_inputs(
                inputs=workspace.config,
                config=self.config,
                device=accelerator.device,
                dataset=dataset,
                predictor=head_type,
            )
            self.prepare_features(workspace, prepared)
            if prepared.input_identity:
                workspace.validate_experiment("inputs", prepared.input_identity)
            return prepared

        logger.info("Running Prepare step...")

        model_id = workspace.model_repo_id or workspace.model_name
        dataset_id = workspace.dataset_repo_id or workspace.dataset_name

        # Load the tokenizer
        logger.info("Loading tokenizer for '%s'...", model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Load the evaluation dataset from HF hub or local cache
        logger.info("Resolving dataset '%s'...", dataset_id)
        dataset = resolve_evaluation_data(
            dataset_id,
            seed=seed,
            validation_fraction=self.config.validation_fraction,
            split=self.config.split,
            dataset=dataset,
            config_name=workspace.config.dataset_config_name,
            revision=workspace.config.dataset_revision,
        )
        shared_context = (
            baseline_context(
                workspace.config,
                dataset,
                baseline_names,
                sequence_column=self.config.sequence_column,
                label_column=self.config.label_column,
                id_column=self.config.id_column,
                split_roles=self.config.split.model_dump(),
            )
            if baseline_names
            else None
        )

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

        # Let rank 0 populate the shared tokenization cache before other ranks use it
        logger.info("Tokenizing dataset...")
        with accelerator.main_process_first():
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
        dataloader_config = DataLoaderConfig(
            batch_size=self.config.batch_size,
            max_tokens_per_batch=self.config.max_tokens_per_batch,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            persistent_workers=self.config.dataloader_persistent_workers,
            seed=seed,
            length_grouped_sampling=self.config.length_grouped_sampling,
        )
        train_dataloader, val_dataloader, test_dataloader = build_dataloaders(
            dataset=dataset,
            collator=collator,
            config=dataloader_config,
        )
        refit_token_dataloaders = {}
        for split in self.config.split.refit_additional_splits:
            if split not in dataset:
                raise ValueError(
                    f"Configured final-refit source split '{split}' was not loaded. "
                    f"Available splits: {sorted(dataset.keys())}."
                )
            split_dataset = DatasetDict({"train": dataset[split]})
            refit_loader, _, _ = build_dataloaders(
                dataset=split_dataset,
                collator=collator,
                config=dataloader_config,
            )
            if refit_loader is None:
                raise ValueError(
                    f"Configured final-refit source split '{split}' is empty."
                )
            refit_token_dataloaders[split] = accelerator.prepare(refit_loader)

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
        active_layerwise_config = (
            layerwise_config
            if layerwise_config is not None and layerwise_config.enabled
            else None
        )
        if active_layerwise_config is not None:
            validate_embedding_cache_compat(
                task.is_ragged, tokenizer, active_layerwise_config.pooling
            )
            if not self.config.embedding_cache.enabled:
                model = LayerwiseProbeModel(
                    model,
                    num_labels,
                    workspace.task_type,
                    active_layerwise_config,
                    tokenizer=tokenizer,
                ).to(accelerator.device)

        # Leave the model raw/unwrapped here: TuneStep needs to set
        # requires_grad on it before wrapping, and accelerator.prepare()
        # must only be called once on the model.
        # Random-trunk and input controls need original tokens even when probes
        # train on cached features, so preserve the token-based loaders here.
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

        # Without caching, Tune/Predict receive token batches and run the model.
        # With caching, they receive cached features and a probe; the raw token
        # test loader remains available for controls that need to rerun the trunk.
        if self.config.embedding_cache.enabled:
            # Replace token batches with cached features; retain the extraction
            # backend so prediction can choose cached or token-based controls.
            (
                probe_model,
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
                refit_token_dataloaders,
                raw_test_dataloader,
                accelerator,
                seed,
                num_labels,
                # Match live probe representations and include those settings
                # in cache identity, rather than using generic cache defaults.
                cache_config=self._layerwise_cache_config(
                    active_layerwise_config, model
                ),
            )
            # Cache extraction needs the original task model; attach the
            # per-layer trainable heads only after its features are prepared.
            model = (
                LayerwiseProbeModel(
                    model,
                    num_labels,
                    workspace.task_type,
                    active_layerwise_config,
                    tokenizer=tokenizer,
                ).to(accelerator.device)
                if active_layerwise_config is not None
                else probe_model
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
            layerwise=(
                LayerwiseArtifacts(token_dataloaders=token_dataloaders)
                if token_dataloaders is not None
                else None
            ),
            refit_token_dataloaders=refit_token_dataloaders,
            refit_random_truncate_by_split={
                split: self.config.random_truncate_by_split.get(split, False)
                for split in refit_token_dataloaders
            },
            split=self.config.split.name,
            baseline_context=shared_context,
            **embedding_extras,
        )

    def _layerwise_cache_config(
        self, layerwise_config: LayerwiseConfig | None, model: nn.Module
    ) -> EmbeddingCacheConfig | None:
        """Cache settings matching live layerwise probe representations."""
        if layerwise_config is None:
            return None
        return self.config.embedding_cache.model_copy(
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
        self,
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
            max_length=self.config.max_length,
            random_truncate=self.config.random_truncate_by_split.get("test", False),
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
            packed=self.config.packed,
            length_grouped_sampling=self.config.length_grouped_sampling,
        )

    def prepare_external_test(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        external: ExternalTestSetConfig,
        *,
        accelerator: Accelerator,
        seed: int,
        layerwise_config: LayerwiseConfig | None = None,
    ) -> PreparedArtifacts:
        """Return *prepared* with its test loaders swapped for an external set.

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
        dataset = resolve_external_data(
            external.dataset_repo_id,
            external.split,
            config_name=external.dataset_config_name,
            revision=external.dataset_revision,
        )
        shared_context = (
            external_baseline_context(
                prepared.baseline_context,
                workspace.config,
                dataset,
                dataset_id=external.dataset_repo_id,
                split=external.split,
                config_name=external.dataset_config_name,
                revision=external.dataset_revision,
                sequence_column=self.config.sequence_column,
                label_column=self.config.label_column,
                id_column=self.config.id_column,
            )
            if prepared.baseline_context
            else None
        )

        task = get_task(workspace.task_type)
        num_labels = task.resolve_num_labels(
            workspace.num_labels, workspace.dataset_name
        )
        task.validate_num_labels(
            dataset, self.config.label_column, num_labels, external.name
        )
        with accelerator.main_process_first():
            dataset = tokenize_dataset(
                dataset,
                tokenizer=prepared.tokenizer,
                task_type=workspace.task_type,
                sequence_column=self.config.sequence_column,
                label_column=self.config.label_column,
                max_length=self.config.max_length,
                id_column=self.config.id_column,
                num_proc=self.config.tokenize_num_proc,
                random_truncate_by_split=self.config.random_truncate_by_split,
                seed=seed,
            )
        _, _, raw_test_dataloader = build_dataloaders(
            dataset=dataset,
            collator=self._build_collator(prepared.tokenizer, workspace.task_type),
            config=DataLoaderConfig(
                batch_size=self.config.batch_size,
                max_tokens_per_batch=self.config.max_tokens_per_batch,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                persistent_workers=self.config.dataloader_persistent_workers,
                seed=seed,
                length_grouped_sampling=self.config.length_grouped_sampling,
            ),
        )
        if raw_test_dataloader is None:
            raise ValueError(
                f"External test set '{external.name}' did not produce a test "
                "dataloader."
            )
        token_test_dataloader = accelerator.prepare(raw_test_dataloader)

        embedding = prepared.embedding
        backend = embedding.backend if embedding is not None else None
        layerwise = (
            replace(
                prepared.layerwise,
                token_dataloaders={
                    **prepared.layerwise.token_dataloaders,
                    "test": token_test_dataloader,
                },
            )
            if prepared.layerwise is not None
            else None
        )
        if backend is None:
            return replace(
                prepared,
                test_dataloader=token_test_dataloader,
                layerwise=layerwise,
                baseline_context=shared_context,
            )
        if backend.cache is None or backend.model_id is None:
            raise ValueError(
                "Embedding-cache external evaluation requires the cache and "
                "model ID recorded by the original Prepare step."
            )

        trunk = backend.trunk
        cache_config = (
            self._layerwise_cache_config(
                layerwise_config
                if layerwise_config and layerwise_config.enabled
                else None,
                trunk,
            )
            or self.config.embedding_cache
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
                    self.config.random_truncate_by_split.get("test", False),
                )
            },
            accelerator=accelerator,
            cache=backend.cache,
            cache_config=cache_config,
            model_id=backend.model_id,
            dataset_id=external_id,
            max_length=self.config.max_length,
            seed=seed,
            batch_size=self.config.batch_size,
            extraction_batch_size=(
                None if self.config.packed else self.config.embedding_batch_size
            ),
            embedding_size=backend.hidden_size,
            label_dtype=task.label_dtype,
            is_ragged=task.is_ragged,
            model_fingerprint=model_fingerprint,
            length_grouped_sampling=self.config.length_grouped_sampling,
        )["test"]
        external_backend = replace(
            backend,
            token_test_dataloader=raw_test_dataloader,
            dataset_id=external_id,
            test_cache_key=self._test_cache_key(
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
            prepared,
            test_dataloader=embedding_dataloader,
            embedding=EmbeddingArtifacts(backend=external_backend),
            layerwise=layerwise,
            baseline_context=shared_context,
        )

    def _apply_embedding_cache(
        self,
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
        num_labels: int,
        cache_config: EmbeddingCacheConfig | None = None,
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
        cache_config = cache_config or self.config.embedding_cache
        task = get_task(workspace.task_type)
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
        # The original token-based test dataloader, kept for PredictStep
        # controls/options that need real tokens or a re-embeddable trunk
        # ("random_trunk", "scrambled_sequences", "store_sequences") that a
        # pooled-embedding-only test_dataloader can no longer provide.
        token_test_dataloader = raw_test_dataloader
        model_fingerprint = weights_fingerprint(model)
        test_random_truncate = self.config.random_truncate_by_split.get("test", False)
        test_cache_key = self._test_cache_key(
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
                **{
                    split: (
                        dataloader,
                        self.config.random_truncate_by_split.get(split, False),
                    )
                    for split, dataloader in refit_token_dataloaders.items()
                },
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
            length_grouped_sampling=self.config.length_grouped_sampling,
        )

        probe_model = EmbeddingProbeHead(
            embedding_size, num_labels, workspace.task_type
        )
        probe_model = probe_model.to(accelerator.device)
        refit_embedding_dataloaders = {}
        for split in refit_token_dataloaders:
            dataloader = embedding_dataloaders[split]
            if dataloader is None:
                raise ValueError(
                    f"Could not build embeddings for final-refit split '{split}'."
                )
            refit_embedding_dataloaders[split] = dataloader
        embedding_extras = {
            "embedding": EmbeddingArtifacts(
                backend=EmbeddingBackend(
                    trunk=model,
                    token_test_dataloader=token_test_dataloader,
                    token_train_dataloader=train_dataloader,
                    token_val_dataloader=val_dataloader,
                    train_random_truncate=self.config.random_truncate_by_split.get(
                        "train", True
                    ),
                    validation_random_truncate=self.config.random_truncate_by_split.get(
                        "validation", False
                    ),
                    extraction_batch_size=None
                    if self.config.packed
                    else self.config.embedding_batch_size,
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
                    length_grouped_sampling=self.config.length_grouped_sampling,
                    test_dataset_fingerprint=getattr(
                        token_test_dataloader.dataset, "_fingerprint", None
                    ),
                )
            ),
            "refit_dataloaders": refit_embedding_dataloaders,
        }
        return (
            probe_model,
            embedding_dataloaders["train"],
            embedding_dataloaders["validation"],
            embedding_dataloaders["test"],
            embedding_extras,
        )

    def prepare_features(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
    ) -> None:
        """Materialize declared fitting inputs under the configured cache policy."""
        task = get_task(workspace.task_type)
        config = self.config.embedding_cache
        prepared.features = {}
        for name, request in task.feature_requests(prepared).items():
            values, fingerprint = load_tensor_cache(
                cache_dir=workspace.embeddings_path
                if config.cache_policy != "off"
                else None,
                namespace=request.namespace,
                identity=request.identity,
                extract=request.extract,
                validate=partial(
                    validate_tensor_cache, shapes=request.shapes, dtype=request.dtype
                ),
                refresh=config.cache_policy == "refresh",
                max_cache_gb=config.max_cache_gb,
            )
            prepared.features[name] = PreparedFeatures(values, fingerprint)
        prepared.input_identity = task.fitting_identity(prepared)

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
