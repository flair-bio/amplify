"""Canonical identities for models, datasets and evaluation outputs.

Identities are JSON-serializable dictionaries hashed with SHA-256 over a canonical
encoding (sorted keys, no NaN). Local checkpoints and local dataset trees are
content-addressed by streaming hashes of their files, so a changed weight or
split file changes the identity without loading tensors into memory.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{40}")
_CHUNK_BYTES = 1 << 20


def canonical_json(payload: Any) -> str:
    """Encode *payload* deterministically; rejects NaN and infinity."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(payload: Any) -> str:
    """Full SHA-256 hex digest of the canonical encoding of *payload*."""
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def predictor_identity(
    *,
    mode: str,
    head: Mapping[str, Any],
    layerwise: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity of the predictor that produced a run.

    Predictor settings (fitting mode, head, layerwise probe) belong to the
    per-run identity, not the workspace identity: sibling predictors share one
    workspace label and dataset/model pair, so binding the workspace to a single
    predictor would wrongly reject the others. The returned ``id`` changes when
    any predictor-defining setting changes.
    """
    body = {
        "schema": SCHEMA_VERSION,
        "mode": mode,
        "head": dict(head),
        "layerwise": dict(layerwise),
    }
    return {"id": digest(body), **body}


def is_immutable_revision(revision: str | None) -> bool:
    """True only for a full 40-character commit SHA (branch names are mutable)."""
    return revision is not None and _COMMIT_SHA.fullmatch(revision) is not None


def validate_identity_label(value: str, kind: str) -> str:
    if not _SAFE_LABEL.fullmatch(value):
        raise ValueError(
            f"{kind} label {value!r} must start with a letter or digit and contain "
            "only letters, digits, '.', '_', or '-'."
        )
    return value


def file_tree_fingerprint(path: str | Path) -> str:
    """SHA-256 over a file or directory tree.

    Each file contributes its relative POSIX path, size and streamed bytes, in
    sorted order, so renamed, added, removed or edited files change the digest.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    files = (
        [root]
        if root.is_file()
        else sorted(item for item in root.rglob("*") if item.is_file())
    )
    hasher = hashlib.sha256()
    for file in files:
        relative = "." if file == root else file.relative_to(root).as_posix()
        hasher.update(f"{relative}\0{file.stat().st_size}\0".encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def is_local_source(repo_id: str) -> bool:
    """Local data/models are absolute or existing directories (see the loaders)."""
    return Path(repo_id).is_absolute() or Path(repo_id).is_dir()


def validate_model_pin(*, repo_id: str | None, revision: str | None) -> None:
    """Offline pin rules for a model: revisions are 40-char SHAs for Hub models only."""
    if revision is None:
        return
    if not is_immutable_revision(revision):
        raise ValueError(
            f"model_revision must be a full 40-character commit SHA, got {revision!r}"
        )
    if repo_id is not None and is_local_source(repo_id):
        raise ValueError("model_revision applies only to Hub models, not local paths")


def validate_dataset_pin(
    *, task_type: str, repo_id: str | None, revision: str | None, subset: str | None
) -> None:
    """Offline pin rules for a dataset (no Hub or data access).

    Revisions must be full commit SHAs. Hub contact data must be pinned because
    the CASP protocol is defined for a fixed commit; local data (``--data-root``)
    takes no revision.
    """
    if revision is not None and not is_immutable_revision(revision):
        raise ValueError(
            f"dataset_revision must be a full 40-character commit SHA, got {revision!r}"
        )
    if repo_id is None:
        return
    local = is_local_source(repo_id)
    if local and revision is not None:
        raise ValueError(
            "dataset_revision applies only to Hub datasets, not local paths"
        )
    if task_type == "contact_prediction":
        if subset is not None:
            raise ValueError("contact prediction does not support dataset subsets")
        if not local and revision is None:
            raise ValueError(
                "Hub contact data requires a pinned 40-character dataset_revision; "
                "supply workspace.dataset_revision=<sha> or local data via --data-root"
            )


def model_identity(
    *, label: str, source: str, requested_revision: str | None
) -> dict[str, Any]:
    """Identity of one model: safe label, source, requested revision, checkpoint.

    Local checkpoint directories are fingerprinted by content. Hub models record
    the requested revision; their resolved content is not known until loaded, so
    ``fingerprint`` stays ``None`` for them.
    """
    validate_identity_label(label, "Model")
    local = Path(source).is_dir()
    if local and requested_revision is not None:
        raise ValueError("model revision applies only to Hub models")
    if requested_revision is not None and not is_immutable_revision(requested_revision):
        raise ValueError(
            "model revision must be a full 40-character commit SHA, "
            f"got {requested_revision!r}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "source": source,
        "local": local,
        "requested_revision": requested_revision,
        "fingerprint": file_tree_fingerprint(source) if local else None,
    }


def dataset_identity(
    *, source: str, subset: str | None, revision: str | None
) -> dict[str, Any]:
    """Identity of one dataset view: source, explicit subset, pinned revision.

    Local directories are content-addressed (the selected subset's tree, or the
    whole directory when no subset is named). Hub sources are pinned only by an
    immutable commit SHA; an unpinned Hub source is recorded as such.
    """
    if subset is not None:
        validate_identity_label(subset, "Dataset subset")
    local_root = Path(source)
    local = local_root.is_dir()
    if local and revision is not None:
        raise ValueError("dataset revision applies only to Hub datasets")
    if revision is not None and not is_immutable_revision(revision):
        raise ValueError(
            f"dataset revision must be a full 40-character commit SHA, got {revision!r}"
        )
    content = None
    if local:
        content = file_tree_fingerprint(
            local_root if subset is None else local_root / subset
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "subset": subset,
        "local": local,
        "revision": revision,
        "pinned": local or is_immutable_revision(revision),
        "content": content,
    }
