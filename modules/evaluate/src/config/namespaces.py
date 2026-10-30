"""Canonical evaluation config namespaces and the legacy-key alias layer.

Authors write canonical namespaces: ``runtime``, ``workspace`` (run/output
fields only), ``dataset``, ``task``, ``model``, ``predictor``,
``representation``, ``training``, ``prediction``, ``scoring``, ``tracking`` and
``baselines``. Legacy spellings (``seed``, ``tf32``, ``wandb``, ``steps.*`` and
the dataset/model/task fields under ``workspace``) remain accepted as aliases.
Each alias is mapped to its canonical key and logged as deprecated.

Resolution is deterministic:

1. Every YAML file and every dotlist override is canonicalized on its own. Two
   spellings of the same setting inside one layer are an error.
2. Canonical layers merge left to right, then dotlist overrides, so the last
   assignment wins regardless of spelling.
3. The merged canonical tree is translated back to the internal legacy shape
   (``EvaluationConfig``), which remains the strict validator.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path as FilePath
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

KeyPath = tuple[str, ...]

# (legacy path, canonical path). A pair applies to the path and everything below
# it, matching whole segments only. Legacy sides are disjoint from canonical sides.
_ALIASES: tuple[tuple[KeyPath, KeyPath], ...] = (
    (("seed",), ("runtime", "seed")),
    (("tf32",), ("runtime", "tf32")),
    (("workspace", "base_path"), ("runtime", "output_root")),
    (("wandb",), ("tracking",)),
    (("steps", "mode"), ("predictor", "mode")),
    (("steps", "layerwise"), ("predictor", "layerwise")),
    (("steps", "prepare"), ("representation",)),
    # Listed before the broader ``steps.tune`` prefix: the head is predictor-owned.
    # The first entry is also the inverse used to translate back to the legacy shape.
    (("steps", "tune", "head"), ("predictor", "head")),
    (("training", "head"), ("predictor", "head")),
    (("steps", "tune"), ("training",)),
    (("steps", "predict"), ("prediction",)),
    (("steps", "score"), ("scoring",)),
    (("workspace", "dataset_name"), ("dataset", "name")),
    (("workspace", "dataset_repo_id"), ("dataset", "repo_id")),
    (("workspace", "dataset_revision"), ("dataset", "revision")),
    (("workspace", "dataset_subset"), ("dataset", "subset")),
    (("workspace", "local_path"), ("dataset", "local_path")),
    (("workspace", "edition"), ("dataset", "edition")),
    (("workspace", "target_type"), ("dataset", "target_type")),
    (("workspace", "label_format"), ("dataset", "label_format")),
    (("workspace", "num_labels"), ("dataset", "num_labels")),
    (("external_test_sets",), ("dataset", "external_test_sets")),
    (("workspace", "task_type"), ("task", "type")),
    (("workspace", "model_name"), ("model", "name")),
    (("workspace", "model_repo_id"), ("model", "repo_id")),
    (("workspace", "model_revision"), ("model", "revision")),
    (("workspace", "model_kwargs"), ("model", "kwargs")),
)
_INVERSE: tuple[tuple[KeyPath, KeyPath], ...] = tuple(
    (canonical, legacy) for legacy, canonical in _ALIASES
)
# Canonical keys with no legacy spelling; only translated for the validator.
_CANONICAL_ONLY: tuple[tuple[KeyPath, KeyPath], ...] = (
    (("runtime", "data_root"), ("data_root",)),
)
# Top-level namespaces whose leaves are used as-is (workspace run/output fields,
# and baselines, which keep their established shape).
_PASSTHROUGH = frozenset({"workspace", "baselines"})
_CANONICAL_ROOTS = (
    frozenset(
        {
            "runtime",
            "dataset",
            "task",
            "model",
            "predictor",
            "representation",
            "training",
            "prediction",
            "scoring",
            "tracking",
        }
    )
    | _PASSTHROUGH
)


def _dotted(path: KeyPath) -> str:
    return ".".join(path)


def _rewrite(
    path: KeyPath, pairs: tuple[tuple[KeyPath, KeyPath], ...]
) -> tuple[KeyPath, bool]:
    for source, target in pairs:
        if path[: len(source)] == source:
            return target + path[len(source) :], True
    return path, False


def _is_known_canonical(path: KeyPath) -> bool:
    if path and path[0] in _PASSTHROUGH:
        return True
    if any(path[: len(source)] == source for source, _ in _CANONICAL_ONLY):
        return True
    return any(path[: len(target)] == target for _, target in _ALIASES)


def _leaves(
    node: Mapping[str, Any], prefix: KeyPath = ()
) -> Iterator[tuple[KeyPath, Any]]:
    """Yield (path, value) for every leaf; an empty mapping is itself a leaf value."""
    for key, value in node.items():
        path = prefix + (str(key),)
        if isinstance(value, Mapping) and value:
            yield from _leaves(value, path)
        else:
            yield path, dict(value) if isinstance(value, Mapping) else value


def _resolve_key(path: KeyPath, where: str) -> tuple[KeyPath, bool]:
    """Return the canonical path for a key, rejecting names outside the schema."""
    canonical, aliased = _rewrite(path, _ALIASES)
    if not aliased and not _is_known_canonical(path):
        raise ValueError(f"{where}: unknown config setting '{_dotted(path)}'")
    return canonical, aliased


def _set(
    tree: dict[str, Any],
    path: KeyPath,
    value: Any,
    origin: dict[KeyPath, str],
    spelled: str,
    where: str,
) -> None:
    node = tree
    for depth, key in enumerate(path[:-1]):
        child = node.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(
                f"{where}: '{_dotted(path[: depth + 1])}' conflicts with a value"
            )
        node = child
    leaf = path[-1]
    if leaf in node:
        if value == {} and isinstance(node[leaf], dict):
            return  # an empty mapping adds no settings beside nested ones
        first = origin.get(path, "?")
        raise ValueError(
            f"{where}: {_dotted(path)} is set both as {first!r} and {spelled!r}; "
            "keep one spelling per setting in a single file or override list"
        )
    node[leaf] = value
    origin[path] = spelled


_WARN_EXAMPLES = 4


def _warn_legacy(where: str, used: list[tuple[str, str]]) -> None:
    if used:
        shown = ", ".join(f"{old} -> {new}" for old, new in used[:_WARN_EXAMPLES])
        more = len(used) - _WARN_EXAMPLES
        suffix = f" and {more} more" if more > 0 else ""
        logger.warning(
            "%s uses %d deprecated legacy config key(s) (e.g. %s%s); use the canonical "
            "namespaces (see modules/evaluate/README.md, 'Config namespaces').",
            where,
            len(used),
            shown,
            suffix,
        )


def canonicalize_layer(raw: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Map one YAML document onto canonical keys, rejecting duplicate spellings."""
    tree: dict[str, Any] = {}
    origin: dict[KeyPath, str] = {}
    used: list[tuple[str, str]] = []
    for path, value in _leaves(raw):
        canonical, aliased = _resolve_key(path, where)
        if aliased:
            used.append((_dotted(path), _dotted(canonical)))
        _set(tree, canonical, value, origin, _dotted(path), where)
    _warn_legacy(where, used)
    return tree


def canonicalize_override(setting: str) -> tuple[KeyPath, str]:
    """Return the canonical path and the original text of one ``key=value`` assignment."""
    key, sep, _ = setting.partition("=")
    if not sep or not key:
        raise ValueError(f"override must be key=value: '{setting}'")
    canonical, _ = _resolve_key(tuple(key.split(".")), f"override '{setting}'")
    return canonical, setting


def canonical_setting(setting: str) -> KeyPath:
    """Return the canonical key path that an override assigns."""
    return canonicalize_override(setting)[0]


def reject_conflicting_settings(settings: Sequence[str]) -> None:
    """Reject two assignments that set the same canonical key (e.g. ``seed`` and ``runtime.seed``)."""
    seen: dict[KeyPath, str] = {}
    for setting in settings:
        key = canonical_setting(setting)
        if key in seen:
            raise ValueError(
                f"conflicting assignments for {_dotted(key)}: {seen[key]!r} and {setting!r}"
            )
        seen[key] = setting


def _canonicalize_overrides(overrides: Sequence[str]) -> list[str]:
    result: list[str] = []
    used: list[tuple[str, str]] = []
    for setting in overrides:
        key, _, value = setting.partition("=")
        canonical, aliased = _resolve_key(
            tuple(key.split(".")), f"override '{setting}'"
        )
        if aliased:
            used.append((key, _dotted(canonical)))
        result.append(f"{_dotted(canonical)}={value}")
    _warn_legacy("command-line overrides", used)
    return result


def _load_layer(path: FilePath | str) -> dict[str, Any]:
    loaded = OmegaConf.load(str(path))
    raw = OmegaConf.to_container(loaded, resolve=False)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config document must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _to_legacy(canonical: Mapping[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    origin: dict[KeyPath, str] = {}
    for path, value in _leaves(canonical):
        legacy, _ = _rewrite(path, _INVERSE + _CANONICAL_ONLY)
        _set(tree, legacy, value, origin, _dotted(path), "merged configuration")
    return tree


def compose_canonical(
    paths: Sequence[str | FilePath], overrides: Sequence[str] | None = None
) -> DictConfig:
    """Compose canonical layers and return the legacy-shaped config for validation."""
    if not paths:
        raise ValueError("At least one config. path must be provided.")
    layers = [canonicalize_layer(_load_layer(p), str(p)) for p in paths]
    merged = OmegaConf.merge(*[OmegaConf.create(layer) for layer in layers])
    if overrides:
        merged = OmegaConf.merge(
            merged, OmegaConf.from_dotlist(_canonicalize_overrides(overrides))
        )
    canonical = OmegaConf.to_container(merged, resolve=False)
    assert isinstance(canonical, dict)
    legacy = _to_legacy({str(k): v for k, v in canonical.items()})
    return OmegaConf.create(legacy)
