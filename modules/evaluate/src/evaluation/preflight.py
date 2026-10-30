"""Execution preflight: pin remote Hub references to immutable commit SHAs.

Real execution calls this module once per job, before the workspace is created,
so output identities bind to the exact commits that are loaded. Dry-run never
imports or calls it, which keeps composition offline. Local directories and
revisions that are already pinned are returned unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, TypeVar

from huggingface_hub import HfApi

from modules.evaluate.src.artifacts.identity import (
    is_immutable_revision,
    is_local_source,
)
from modules.evaluate.src.config.evaluation import EvaluationConfig
from modules.evaluate.src.dataset.dataloader import ExternalTestSetConfig
from modules.evaluate.src.dataset.workspace import (
    DatasetWorkspaceConfig,
    EvaluationWorkspaceConfig,
)

__all__ = [
    "ResolveFn",
    "pin_dataset_views",
    "pin_evaluation_config",
    "pin_model_revision",
    "resolve_hub_commit",
]

logger = logging.getLogger(__name__)

ResolveFn = Callable[[str, Literal["model", "dataset"]], str]
_WorkspaceT = TypeVar("_WorkspaceT", bound=DatasetWorkspaceConfig)


def _resolver(resolve: ResolveFn | None) -> ResolveFn:
    """The injected resolver, else the Hub resolver looked up at call time."""
    return resolve if resolve is not None else resolve_hub_commit


def resolve_hub_commit(repo_id: str, kind: Literal["model", "dataset"]) -> str:
    """Return the commit SHA the Hub currently serves for *repo_id*'s default ref.

    Only called for remote sources without a pinned revision. Failures carry
    the explicit pinning remedy, because an unresolvable repo cannot be bound
    to immutable inputs.
    """
    api = HfApi()
    try:
        info = api.model_info(repo_id) if kind == "model" else api.dataset_info(repo_id)
    except Exception as error:
        raise RuntimeError(
            f"Could not resolve the default revision of {kind} {repo_id!r} from the "
            f"Hub ({error}). Pin it with workspace.{kind}_revision=<40-character commit "
            "SHA>, or supply local data/checkpoints."
        ) from error
    sha = info.sha
    if sha is None or not is_immutable_revision(sha):
        raise RuntimeError(
            f"The Hub returned no 40-character commit SHA for {kind} {repo_id!r}."
        )
    logger.info("Pinned %s %s to commit %s", kind, repo_id, sha)
    return sha


def _hub_repo(repo_id: str | None) -> str | None:
    """The repo id when it names a Hub source, otherwise None."""
    if repo_id is None or is_local_source(repo_id):
        return None
    return repo_id


def pin_model_revision(
    workspace: EvaluationWorkspaceConfig, *, resolve: ResolveFn | None = None
) -> EvaluationWorkspaceConfig:
    """Pin the Hub model of *workspace* when no revision was requested."""
    repo = _hub_repo(workspace.model_repo_id)
    if repo is None or workspace.model_revision is not None:
        return workspace
    return workspace.model_copy(
        update={"model_revision": _resolver(resolve)(repo, "model")}
    )


def pin_dataset_views(
    workspace: _WorkspaceT,
    external_test_sets: list[ExternalTestSetConfig],
    *,
    resolve: ResolveFn | None = None,
) -> tuple[_WorkspaceT, list[ExternalTestSetConfig]]:
    """Pin the main and external Hub datasets that were left unpinned."""
    resolver = _resolver(resolve)
    repo = _hub_repo(workspace.dataset_repo_id)
    if repo is not None and workspace.dataset_revision is None:
        workspace = workspace.model_copy(
            update={"dataset_revision": resolver(repo, "dataset")}
        )
    pinned: list[ExternalTestSetConfig] = []
    for external in external_test_sets:
        external_repo = _hub_repo(external.dataset_repo_id)
        if external_repo is not None and external.dataset_revision is None:
            external = external.model_copy(
                update={"dataset_revision": resolver(external_repo, "dataset")}
            )
        pinned.append(external)
    return workspace, pinned


def pin_evaluation_config(
    config: EvaluationConfig, *, resolve: ResolveFn | None = None
) -> EvaluationConfig:
    """Return *config* with every unpinned remote model/dataset SHA resolved."""
    workspace = pin_model_revision(config.workspace, resolve=resolve)
    workspace, externals = pin_dataset_views(
        workspace, config.external_test_sets, resolve=resolve
    )
    return config.model_copy(
        update={"workspace": workspace, "external_test_sets": externals}
    )
