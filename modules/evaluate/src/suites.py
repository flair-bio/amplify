"""Expand evaluation suites into explicit task/dataset/predictor jobs.

A suite declares a task, the datasets it evaluates and optional predictors.
Expansion is pure configuration work: it reads YAML and validates references.
It never loads checkpoints, reads datasets or writes results; each job's
config is composed later by the same loader used for single runs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.evaluate.src.tasks import available_tasks

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
SUITE_DIR = CONFIG_ROOT / "suites"
MODEL_COLLECTION_DIR = CONFIG_ROOT / "model_collections"
MODEL_DIR = CONFIG_ROOT / "models"
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SuiteError(ValueError):
    """Raised for invalid suite, selector or model-collection references."""


class EvaluationEntry(BaseModel):
    """One dataset within a suite; ``predictors=None`` inherits the suite list."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    predictors: list[str] | None = None
    overrides: list[str] = Field(default_factory=list)


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str
    task: str | None = None
    predictors: list[str] = Field(default_factory=list)
    includes: list[str] = Field(default_factory=list)
    evaluations: list[EvaluationEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_shape(self) -> SuiteSpec:
        if self.includes and (self.evaluations or self.task or self.predictors):
            raise ValueError(
                f"suite '{self.suite}' is an aggregate: it may only declare 'includes'"
            )
        if self.evaluations and self.task is None:
            raise ValueError(f"suite '{self.suite}' declares evaluations but no task")
        if self.task is not None and self.task not in available_tasks():
            raise ValueError(
                f"suite '{self.suite}' has unknown task '{self.task}'. "
                f"Available: {available_tasks()}."
            )
        return self


@dataclass(frozen=True)
class EvaluationJob:
    """A task/dataset/predictor evaluation; model choice is applied separately."""

    suite: str
    task: str
    dataset: str
    predictor: str | None
    overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def evaluation_id(self) -> str:
        return f"{self.task}/{self.dataset}/{self.predictor or 'default'}"

    def config_paths(self, model_path: Path | None = None) -> list[Path]:
        """Overlay order: base config, model, task, dataset, predictor.

        Explicit user overrides are applied by the caller after these files.
        """
        paths = [CONFIG_ROOT / "config.yaml"]
        if model_path is not None:
            paths.append(model_path)
        paths.append(CONFIG_ROOT / "tasks" / f"{self.task}.yaml")
        paths.append(CONFIG_ROOT / "datasets" / f"{self.dataset}.yaml")
        if self.predictor is not None:
            paths.append(CONFIG_ROOT / "predictors" / f"{self.predictor}.yaml")
        return paths


def _suite_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"}:
        return path
    return SUITE_DIR / f"{name_or_path}.yaml"


def _read_suite(name_or_path: str | Path) -> SuiteSpec:
    path = _suite_path(name_or_path)
    if not path.is_file():
        raise SuiteError(f"suite '{name_or_path}' not found at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return SuiteSpec.model_validate(raw)
    except ValueError as error:
        raise SuiteError(f"invalid suite file {path}: {error}") from error


def expand_suite(
    name_or_path: str | Path,
    _stack: tuple[str, ...] = (),
) -> list[EvaluationJob]:
    """Expand a suite and its includes (declared order) into jobs.

    Include cycles and duplicate evaluation IDs are errors, never silent merges.
    """
    spec = _read_suite(name_or_path)
    if spec.suite in _stack:
        cycle = " -> ".join((*_stack, spec.suite))
        raise SuiteError(f"suite include cycle: {cycle}")
    stack = (*_stack, spec.suite)
    jobs: list[EvaluationJob] = []
    for included in spec.includes:
        jobs.extend(expand_suite(included, stack))
    for entry in spec.evaluations:
        if not (CONFIG_ROOT / "datasets" / f"{entry.dataset}.yaml").is_file():
            raise SuiteError(f"suite '{spec.suite}': unknown dataset '{entry.dataset}'")
        predictors: list[str | None] = list(
            spec.predictors if entry.predictors is None else entry.predictors
        ) or [None]
        for predictor in predictors:
            if predictor is not None and not (
                CONFIG_ROOT / "predictors" / f"{predictor}.yaml"
            ).is_file():
                raise SuiteError(
                    f"suite '{spec.suite}': unknown predictor '{predictor}'"
                )
            jobs.append(
                EvaluationJob(
                    suite=spec.suite,
                    task=spec.task,  # type: ignore[arg-type]
                    dataset=entry.dataset,
                    predictor=predictor,
                    overrides=tuple(entry.overrides),
                )
            )
    if not _stack:
        seen: set[str] = set()
        for job in jobs:
            if job.evaluation_id in seen:
                raise SuiteError(f"duplicate evaluation '{job.evaluation_id}'")
            seen.add(job.evaluation_id)
    return jobs


def select_jobs(
    jobs: Sequence[EvaluationJob],
    predictor: str | None = None,
    dataset: str | None = None,
) -> list[EvaluationJob]:
    """Filter jobs before any model or data is loaded; no match is an error."""
    selected = [
        job
        for job in jobs
        if (predictor is None or job.predictor == predictor)
        and (dataset is None or job.dataset == dataset)
    ]
    if not selected:
        raise SuiteError(
            f"no jobs match predictor={predictor!r} dataset={dataset!r} in this suite"
        )
    return selected


def resolve_model_paths(model: str | Path) -> list[Path]:
    """Resolve a model name or explicit YAML path to a model config file.

    Local checkpoint directories are not supported yet and fail explicitly.
    """
    text = str(model)
    if text.endswith((".yaml", ".yml")):
        path = Path(text)
    elif "/" in text or Path(text).is_dir():
        raise SuiteError(
            f"model '{text}' looks like a checkpoint path; provide a model config "
            "name or YAML (checkpoint directories are not supported yet)"
        )
    else:
        path = MODEL_DIR / f"{text}.yaml"
    if not path.is_file():
        raise SuiteError(f"model config '{text}' not found at {path}")
    return [path]


def resolve_model_collection(name_or_path: str | Path) -> list[Path]:
    """Return ordered, unique model config paths from a model collection file."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = MODEL_COLLECTION_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise SuiteError(f"model collection '{name_or_path}' not found at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    models = raw.get("models")
    if not isinstance(models, list) or not models:
        raise SuiteError(f"model collection {path} must list 'models'")
    labels = [str(model) for model in models]
    if len(set(labels)) != len(labels):
        raise SuiteError(f"model collection {path} has duplicate model entries")
    resolved: list[Path] = []
    for label in labels:
        resolved.extend(resolve_model_paths(label))
    return resolved


def model_label(path: Path) -> str:
    """Filesystem-safe model label derived from a model config file name."""
    label = path.stem
    if not _SAFE_LABEL.match(label):
        raise SuiteError(f"unsafe model label '{label}' from {path}")
    return label
