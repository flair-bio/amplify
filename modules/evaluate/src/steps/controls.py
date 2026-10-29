"""Build predictions for model-based inference ablations."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from accelerate import Accelerator
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, DistributedSampler

from modules.evaluate.src.contracts.artifacts import PreparedArtifacts
from modules.evaluate.src.contracts.config import BASELINE_NAMES, EmbeddingCacheConfig
from modules.evaluate.src.dataset.embeddings import (
    EmbeddingDataset,
    build_embedding_dataloaders,
    embedding_collate_fn,
    find_embedding_dataset,
)
from modules.evaluate.src.dataset.workspace import EvaluationWorkspace
from modules.evaluate.src.model.controls import (
    RANDOM_TRUNK_CONTROL,
    RANDOMIZED_HEAD_CONTROLS,
    RANDOMIZED_TRUNK_CONTROLS,
    SCRAMBLED_SEQUENCES_CONTROL,
    UNTRAINED_CONTROL,
    build_control_model,
    build_random_trunk,
)
from modules.evaluate.src.model.embeddings import extract_embeddings
from modules.evaluate.src.model.heads import EmbeddingProbeHead, run_classical_inference
from modules.evaluate.src.model.inference import (
    run_inference,
    run_local_head_inference,
)
from modules.evaluate.src.model.layerwise import LayerwiseProbeModel
from modules.evaluate.src.steps.prepare import PrepareStep
from modules.evaluate.src.steps.tune import TuneStep
from modules.evaluate.src.tasks import get_task
from modules.evaluate.src.trainers.layerwise import LayerwiseProbeTrainer
from modules.evaluate.src.utils.io import write_json

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from accelerate import Accelerator

    from modules.evaluate.src.contracts.artifacts import PreparedArtifacts
    from modules.evaluate.src.contracts.config import PrepareConfig, TuneConfig
    from modules.evaluate.src.dataset.workspace import EvaluationWorkspace


class ControlsStep:
    """Prepare and fit independent controls without changing the primary model."""

    def __init__(self, config: TuneConfig, prepare_config: PrepareConfig) -> None:
        self.config = config
        self.prepare_config = prepare_config

    def run(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        seed: int = 710019,
        accelerator: Accelerator | None = None,
        controls: Sequence[str] = (),
        *,
        fitted: bool = True,
    ) -> PreparedArtifacts:
        """Materialize requested controls; inference-only ablations stay in Predict."""
        if not controls:
            return prepared
        if accelerator is None:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.config.accumulation_steps
            )
        if prepared.layerwise is not None:
            if {"random_trunk", "random_both"} & set(controls):
                inner = accelerator.unwrap_model(prepared.model)
                if not isinstance(inner, LayerwiseProbeModel):
                    raise TypeError("Layerwise controls require a layerwise predictor.")
                prepared.layerwise.random_trunk_model = LayerwiseProbeTrainer(
                    self.config
                ).fit_random_trunk_control(
                    workspace.config,
                    prepared,
                    inner,
                    accelerator,
                    seed,
                    train_heads="random_trunk" in controls,
                )
            return prepared
        if fitted and prepared.embedding is not None and "random_trunk" in controls:
            self._fit_cached_random_trunk(workspace, prepared, seed, accelerator)
        task = get_task(workspace.task_type, workspace.num_labels)
        for name, inputs in task.independent_controls(
            prepared,
            controls,
            seed=seed,
            device=accelerator.device,
        ).items():
            target = workspace.control_workspace(name)
            target.setup_directories()
            control = replace(
                prepared, model=inputs.model, features={}, input_identity={}
            )
            PrepareStep(self.prepare_config).prepare_features(target, control)
            result = task.fit(
                control, self.config, target.config, accelerator, seed=inputs.seed
            )
            settings = task.control_settings(inputs, result)
            if accelerator.is_main_process:
                inputs.serialization_model.save_pretrained(target.model_path)
                if prepared.tokenizer is not None:
                    prepared.tokenizer.save_pretrained(target.model_path)
                for filename in result.statistics:
                    write_json(target.stats_path / filename, settings)
            accelerator.wait_for_everyone()
            prepared.controls.independent_models[name] = result.model
        return prepared

    def _fit_cached_random_trunk(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        seed: int,
        accelerator: Accelerator,
    ) -> None:
        """Build cached random-trunk features and fit an independent control."""
        backend = prepared.embedding.backend if prepared.embedding is not None else None
        if backend is None:
            return
        if backend.token_train_dataloader is None or backend.cache is None:
            raise ValueError(
                "Cached random_trunk tuning requires retained train tokens and an embedding cache backend."
            )
        if (
            backend.model_id is None
            or backend.dataset_id is None
            or backend.max_length is None
        ):
            raise ValueError(
                "Cached random_trunk tuning requires model/dataset cache metadata."
            )
        task = get_task(workspace.task_type, workspace.num_labels)
        trunk = build_random_trunk(accelerator.unwrap_model(backend.trunk), seed)
        splits = {
            "train": (backend.token_train_dataloader, backend.train_random_truncate),
            "validation": (
                backend.token_val_dataloader,
                backend.validation_random_truncate,
            ),
            **{
                name: (
                    loader,
                    prepared.refit_random_truncate_by_split.get(name, False),
                )
                for name, loader in prepared.refit_token_dataloaders.items()
            },
        }
        loaders = build_embedding_dataloaders(
            model=trunk,
            splits=splits,
            accelerator=accelerator,
            cache=backend.cache,
            cache_config=EmbeddingCacheConfig(
                enabled=True,
                pooling=backend.pooling,
                layers=backend.layers,
                dtype=backend.dtype,
                autocast_dtype=backend.autocast_dtype,
                normalize_before_pooling=backend.normalize_before_pooling,
                cache_policy=backend.cache_policy,
            ),
            model_id=backend.model_id,
            dataset_id=backend.dataset_id,
            max_length=backend.max_length,
            seed=seed,
            batch_size=self.config.batch_size
            or getattr(prepared.train_dataloader, "batch_size", None)
            or getattr(
                getattr(prepared.train_dataloader, "batch_sampler", None),
                "batch_size",
                None,
            )
            or 128,
            extraction_batch_size=backend.extraction_batch_size,
            embedding_size=backend.hidden_size,
            label_dtype=task.label_dtype,
            is_ragged=task.is_ragged,
            length_grouped_sampling=backend.length_grouped_sampling,
        )
        control_refit = {}
        for name in prepared.refit_token_dataloaders:
            loader = loaders[name]
            if loader is None:
                raise ValueError(
                    f"Could not build random-trunk embeddings for final-refit split '{name}'."
                )
            control_refit[name] = loader
        control_prepared = replace(
            prepared,
            model=EmbeddingProbeHead(
                backend.hidden_size,
                task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
                workspace.task_type,
                probe_hidden_size=self.config.head.hidden_size,
                probe_hidden_depth=self.config.head.hidden_depth,
            ).to(accelerator.device),
            train_dataloader=loaders["train"],
            val_dataloader=loaders["validation"],
            test_dataloader=None,
            refit_dataloaders=control_refit,
            embedding=None,
            controls=type(prepared.controls)(),
        )
        control_workspace = workspace.control_workspace("random_trunk")
        control_config = self.config.model_copy(update={"variance_seeds": []})
        control_result = task.fit(
            control_prepared,
            control_config,
            control_workspace.config,
            accelerator,
            seed=seed,
            mode="tune_probe",
        )
        control_prepared.model = control_result.model
        TuneStep(self.config).publish_fit(
            control_workspace, control_prepared, control_result, accelerator
        )
        prepared.controls.random_trunk_model = control_result.model
        prepared.controls.random_trunk_seed = control_result.seed


class ControlBuilder:
    """Build predictions for model-based inference ablations.

    Expensive trunk features are reused through the embedding cache;
    predictions are always produced by the current head in memory."""

    def __init__(
        self,
        collect_probabilities: bool,
        embedding_batch_size: int | None = None,
    ) -> None:
        self.collect_probabilities = collect_probabilities
        self.embedding_batch_size = embedding_batch_size

    def build(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[pl.DataFrame, Path | None]:
        """Dispatch to the right control-building strategy.

        The returned path is always ``None``. Control predictions depend on
        the fitted head and are deliberately not cached; only trunk
        embeddings are reusable across runs.
        """
        if control in BASELINE_NAMES:
            raise ValueError(
                f"'{control}' is a model-independent baseline; build it with "
                "the baseline service, not ControlBuilder."
            )
        started = monotonic()
        if accelerator.is_main_process:
            logger.info("Control '%s' prediction started", control)
        result = self._get_or_build_control(
            control, workspace, prepared, predictions_df, accelerator, seed
        )
        if accelerator.is_main_process:
            logger.info(
                "Control '%s' prediction finished in %.1fs (%d rows)",
                control,
                monotonic() - started,
                result[0].height,
            )
        return result

    def _get_or_build_control(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        predictions_df: pl.DataFrame,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[pl.DataFrame, Path | None]:
        """Build a model-based control from the current head and embeddings."""
        is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
        using_embeddings = prepared.embedding is not None
        if using_embeddings:
            ids, predictions, probabilities, sequences = (
                self._run_embedding_control_inference(
                    control, workspace, prepared, accelerator, seed
                )
            )
        else:
            model = self._build_control_model(
                control, workspace, prepared, accelerator, seed
            )
            test_dataloader = prepared.test_dataloader
            test_dataset = getattr(test_dataloader, "dataset", None)
            progress_total_batches = (
                len(test_dataloader) if test_dataloader is not None else None
            )
            batch_transform = (
                get_task(workspace.task_type).make_sequence_scrambler(
                    prepared.tokenizer, seed
                )
                if is_scrambled_sequences
                else None
            )
            ids, predictions, _, probabilities, sequences = run_inference(
                model,
                test_dataloader,
                workspace.task_type,
                accelerator,
                collect_probabilities=self.collect_probabilities,
                batch_transform=batch_transform,
                tokenizer=prepared.tokenizer if is_scrambled_sequences else None,
                progress_label=f"Control '{control}' prediction",
                progress_total_batches=progress_total_batches,
                progress_total_rows=(
                    len(test_dataset) if test_dataset is not None else None
                ),
                progress_unit=get_task(workspace.task_type).prediction_progress_unit,
            )
        control_data: dict[str, list[Any]] = {"id": ids, "prediction": predictions}
        if probabilities is not None:
            control_data["probability"] = probabilities
        if sequences is not None:
            control_data["sequence"] = sequences

        control_df = pl.DataFrame(control_data)
        if (
            predictions_df["id"].n_unique() < predictions_df.height
            or control_df["id"].n_unique() < control_df.height
        ):
            # Masked-residue and ragged-cache rows repeat source ids. Joining
            # on those ids would multiply rows, so require the same fixed
            # test-set order before aligning predictions positionally.
            control_ids = control_df["id"]
            if (
                control_ids.len() != predictions_df.height
                or (control_ids != predictions_df["id"]).any()
            ):
                raise RuntimeError(
                    f"Control '{control}' produced {control_ids.len()} rows in "
                    f"a different order/count than the {predictions_df.height} "
                    "test rows; controls with repeated ids must iterate the test set in "
                    "the same order as the trained model's predictions."
                )
            aligned = predictions_df.select("id").with_columns(control_df.drop("id"))
            return aligned, None

        aligned = predictions_df.select("id").join(
            control_df, on="id", how="left", maintain_order="left", validate="1:1"
        )
        if aligned["prediction"].null_count() > 0:
            raise RuntimeError(
                f"Control '{control}' predictions could not be aligned to all "
                "test example ids; check that the control and test dataloaders "
                "use the same ids."
            )

        return aligned, None

    def _run_embedding_control_inference(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> tuple[list[Any], list[Any], list[Any] | None, list[str] | None]:
        """Rebuild a control's predictions when
        steps.prepare.embedding_cache is enabled: prepared.test_dataloader
        then yields pooled embeddings (not tokens), and prepared.model isn't
        a full HF model with a `.base_model`/`._init_weights` to
        reinitialize the way :meth:`_build_control_model` does.

        The trunk is always frozen/pretrained for embedding-cache runs (a
        cross-step validator in EvaluationConfig requires tune_probe mode),
        so "untrained"'s trunk is identical to the trained model's trunk --
        only the head differs.
        """
        torch.manual_seed(seed)
        assert prepared.embedding is not None
        backend = prepared.embedding.backend
        is_ragged = get_task(workspace.task_type).is_ragged
        # A fitted sklearn/xgboost estimator is never DDP-wrapped (classical
        # heads skip accelerator.prepare() entirely), and isn't a
        # torch.nn.Module, so accelerator.unwrap_model() would raise.
        is_classical_head = isinstance(prepared.model, BaseEstimator)
        trained_head = (
            prepared.model
            if is_classical_head
            else accelerator.unwrap_model(prepared.model)
        )
        if control == RANDOM_TRUNK_CONTROL:
            if prepared.controls.random_trunk_model is None:
                raise ValueError(
                    "Cached random_trunk requires a fitted probe; pass the predict "
                    "controls to ControlsStep.run before prediction."
                )
            if prepared.controls.random_trunk_seed != seed:
                raise ValueError(
                    "Cached random_trunk prediction seed must match its tuning seed."
                )
            trained_head = prepared.controls.random_trunk_model
        needs_fresh_head = control in RANDOMIZED_HEAD_CONTROLS
        if needs_fresh_head and not isinstance(trained_head, EmbeddingProbeHead):
            raise ValueError(
                f"control='{control}' requires a randomly re-initializable "
                "head, which isn't defined for a classical (non-'mlp') "
                "head_type. Use 'random_trunk', 'scrambled_sequences', or "
                "'scrambled_labels' instead (this should have been caught "
                "by EvaluationConfig validation)."
            )

        if control in RANDOMIZED_TRUNK_CONTROLS:
            trained_trunk = accelerator.unwrap_model(backend.trunk)
            trunk = build_random_trunk(trained_trunk, seed)
        else:
            # "untrained"/"random_head": trunk unchanged, only the head
            # differs. "scrambled_sequences": trunk unchanged, only the
            # (scrambled) input differs.
            trunk = backend.trunk

        is_scrambled_sequences = control == SCRAMBLED_SEQUENCES_CONTROL
        embeddings, ids, labels = self._get_or_build_control_embeddings(
            control,
            workspace,
            prepared,
            accelerator,
            seed,
            trunk,
            is_scrambled_sequences,
        )
        sequences = (
            self._decode_scrambled_sequences(workspace, prepared, accelerator, seed)
            if is_scrambled_sequences
            else None
        )

        if needs_fresh_head:
            head_model = EmbeddingProbeHead(
                trained_head.embedding_size,
                trained_head.num_labels,
                workspace.task_type,
                probe_hidden_size=trained_head.probe_hidden_size,
                probe_hidden_depth=trained_head.probe_hidden_depth,
                standardize_features=trained_head.standardize_features,
            ).to(accelerator.device)
            if trained_head.standardize_features:
                head_model.feature_mean.copy_(trained_head.feature_mean)
                head_model.feature_std.copy_(trained_head.feature_std)
        else:
            head_model = trained_head

        label_dtype = (
            torch.float if workspace.task_type == "sequence_regression" else torch.long
        )
        dataset = EmbeddingDataset(embeddings, ids, labels.to(label_dtype))
        if prepared.test_dataloader is None:
            raise ValueError("A test dataloader is required for local inference.")
        batch_size = (
            self.embedding_batch_size or prepared.test_dataloader.batch_size or 16
        )

        if is_classical_head:
            # The estimator isn't distributed, so every rank runs the same
            # full (unsharded) data through it locally instead of the
            # DistributedSampler + gather_for_metrics path below.
            num_labels = get_task(workspace.task_type).resolve_num_labels(
                workspace.num_labels, workspace.dataset_name
            )
            batches = (
                {
                    "inputs_embeds": dataset.embeddings[start : start + batch_size],
                    "labels": dataset.labels[start : start + batch_size],
                    "id": dataset.ids[start : start + batch_size],
                }
                for start in range(0, len(dataset), batch_size)
            )
            control_ids, predictions, _, probabilities = run_classical_inference(
                head_model,
                batches,
                workspace.task_type,
                num_labels,
                collect_probabilities=self.collect_probabilities,
                progress_label=(
                    f"Control '{control}' prediction"
                    if accelerator.is_main_process
                    else None
                ),
            )
            return control_ids, predictions, probabilities, sequences

        if is_ragged:
            # Ragged/token-level tasks have many rows sharing the same
            # sequence id (one row per valid token), so the id-based dedup
            # _run_local_head_inference uses to strip DistributedSampler
            # padding would collapse them all down to one row per id. Every
            # rank already has the full (unsharded) dataset here, so just
            # run it locally with no sampler/gather instead.
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
            )
            control_ids, predictions, probabilities = run_local_head_inference(
                head_model,
                dataloader,
                workspace.task_type,
                self.collect_probabilities,
                accelerator,
                gather_across_processes=False,
                progress_label=f"Control '{control}' prediction",
            )
            # Match run_local_embedding_inference's per-row shape for ragged
            # tasks: each row is one valid token wrapped in a single-element
            # list (not a bare scalar), which is what ScoreStep's
            # flatten_for_metrics (itertools.chain.from_iterable) expects.
            predictions = [[prediction] for prediction in predictions]
            return control_ids, predictions, probabilities, sequences

        # Not accelerator.prepare()'d: extract_embeddings() above already
        # gathered the full test set identically onto every rank, so a
        # second accelerator.gather_for_metrics() (as run_inference() would
        # do on a distributed-sampled dataloader) would duplicate every
        # prediction once per rank instead of returning it once.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(
                dataset,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                shuffle=False,
            ),
            collate_fn=lambda batch: embedding_collate_fn(batch, label_dtype),
        )
        control_ids, predictions, probabilities = run_local_head_inference(
            head_model,
            dataloader,
            workspace.task_type,
            self.collect_probabilities,
            accelerator,
            progress_label=f"Control '{control}' prediction",
        )
        return control_ids, predictions, probabilities, sequences

    def _get_or_build_control_embeddings(
        self,
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
        trunk: Any,
        is_scrambled_sequences: bool,
    ) -> tuple[torch.Tensor, list[Any], torch.Tensor]:
        assert prepared.embedding is not None
        backend = prepared.embedding.backend
        is_ragged = get_task(workspace.task_type).is_ragged
        input_transform = (
            f"scramble_sequences:v1:seed={seed}" if is_scrambled_sequences else None
        )
        cache_key = backend.test_embedding_key()
        if control in RANDOMIZED_TRUNK_CONTROLS or is_scrambled_sequences:
            cache_key = backend.test_embedding_key(
                trunk=trunk,
                seed=seed,
                input_transform=input_transform,
            )

        cache = backend.cache
        if (
            cache is not None
            and cache_key is not None
            and backend.cache_policy == "reuse"
            and cache.exists("test", cache_key)
        ):
            started = monotonic()
            if accelerator.is_main_process:
                logger.info(
                    "Loading cached '%s' control embeddings from disk...", control
                )
            embeddings, ids, labels = cache.read("test", cache_key)
            if accelerator.is_main_process:
                logger.info(
                    "Loaded control '%s' embeddings in %.1fs (%d rows)",
                    control,
                    monotonic() - started,
                    len(ids),
                )
            return embeddings, ids, labels

        cached_dataset = find_embedding_dataset(prepared.test_dataloader)
        if input_transform is None and control not in RANDOMIZED_TRUNK_CONTROLS:
            if cached_dataset is None:
                raise ValueError(
                    "Embedding controls require cached test embeddings or a "
                    "token-based test dataloader."
                )
            embeddings = cached_dataset.embeddings
            ids = cached_dataset.ids
            labels = torch.as_tensor(cached_dataset.labels)
        else:
            batch_transform = (
                get_task(workspace.task_type).make_sequence_scrambler(
                    prepared.tokenizer, seed
                )
                if is_scrambled_sequences
                else None
            )
            started = monotonic()
            if accelerator.is_main_process:
                logger.info("Control '%s' embedding extraction started", control)
            embeddings, ids, labels, _ = extract_embeddings(
                trunk,
                backend.token_test_dataloader,
                accelerator,
                backend.pooling,
                backend.dtype,
                layers=backend.layers or [-1],
                autocast_dtype={
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }[backend.autocast_dtype],
                normalize_before_pooling=backend.normalize_before_pooling,
                batch_transform=batch_transform,
                is_ragged=is_ragged,
                # token_test_dataloader yields the full dataset on every
                # rank already (see its docstring); gathering it again here
                # would double every row per additional rank.
                gather_across_processes=False,
                progress_label=f"Control '{control}' embedding extraction",
            )
            if accelerator.is_main_process:
                logger.info(
                    "Control '%s' embedding extraction finished in %.1fs (%d rows)",
                    control,
                    monotonic() - started,
                    len(ids),
                )
            if (
                cache is not None
                and cache_key is not None
                and backend.cache_policy != "off"
                and not is_ragged
                and cache.fits_budget(
                    embeddings.shape[0], embeddings.shape[-1], backend.dtype
                )
                and accelerator.is_main_process
            ):
                started = monotonic()
                logger.info("Persisting control '%s' embeddings started", control)
                cache.write("test", cache_key, embeddings, ids, labels)
                logger.info(
                    "Persisting control '%s' embeddings finished in %.1fs",
                    control,
                    monotonic() - started,
                )
            # Only the main process writes the cache file above; without this
            # barrier, a later control that reuses this same cache_key could
            # have other ranks see cache.exists()==False (still racing ahead
            # of the write) while the main process, now past the write, sees
            # cache.exists()==True and takes the no-collective cache-hit
            # return path instead of joining the other ranks' extract_embeddings
            # call -- an inter-rank divergence that hangs the collective forever.
            accelerator.wait_for_everyone()
        return embeddings, ids, labels

    def _decode_scrambled_sequences(
        self,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> list[str]:
        assert prepared.embedding is not None
        if prepared.tokenizer is None:
            raise ValueError("Decoding scrambled sequences requires a tokenizer.")
        transform = get_task(workspace.task_type).make_sequence_scrambler(
            prepared.tokenizer, seed
        )
        all_sequences: list[str] = []
        for batch in prepared.embedding.backend.token_test_dataloader:
            batch = dict(batch)
            batch.pop("id", None)
            batch.pop("labels", None)
            transformed = transform(batch)
            # token_test_dataloader yields the full dataset on every rank
            # already (see its docstring): decode locally instead of
            # gathering, which would duplicate every row once per rank.
            all_sequences.extend(
                prepared.tokenizer.batch_decode(
                    transformed["input_ids"].cpu().tolist(), skip_special_tokens=True
                )
            )
        return all_sequences

    @staticmethod
    def _build_control_model(
        control: str,
        workspace: EvaluationWorkspace,
        prepared: PreparedArtifacts,
        accelerator: Accelerator,
        seed: int,
    ) -> Any:
        pretrained_model = None
        if control == UNTRAINED_CONTROL:
            task = get_task(workspace.task_type)
            torch.manual_seed(seed)
            pretrained_model = task.build_model(
                workspace.model_repo_id or workspace.model_name,
                task.resolve_num_labels(workspace.num_labels, workspace.dataset_name),
                workspace.model_kwargs,
            )
        return build_control_model(
            control,
            accelerator.unwrap_model(prepared.model),
            seed,
            pretrained_model,
        )
