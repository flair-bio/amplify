"""Narrow compatibility shims: src/dataset_sourcing forwards to scripts sourcing.

The canonical implementation lives in ``scripts/dataset_sourcing/shared``. The
shims exist only for the protected ProteinGym script's imports. Runtime
evaluation packages must never depend on sourcing code.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

SHIMS = {
    "config": "modules.evaluate.scripts.dataset_sourcing.shared.config",
    "downloads": "modules.evaluate.scripts.dataset_sourcing.shared.downloads",
    "hf_io": "modules.evaluate.scripts.dataset_sourcing.shared.hf_io",
    "outputs": "modules.evaluate.scripts.dataset_sourcing.shared.outputs",
    "pipeline": "modules.evaluate.scripts.dataset_sourcing.shared.pipeline",
    "preprocess": "modules.evaluate.scripts.dataset_sourcing.shared.preprocess",
}
REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("shim", sorted(SHIMS))
def test_shim_reexports_the_canonical_objects(shim: str) -> None:
    compat = importlib.import_module(f"modules.evaluate.src.dataset_sourcing.{shim}")
    canonical = importlib.import_module(SHIMS[shim])
    assert compat.__all__, f"{shim} shim must declare its exports"
    for name in compat.__all__:
        assert getattr(compat, name) is getattr(canonical, name), name


def test_runtime_packages_do_not_import_sourcing() -> None:
    code = (
        "import sys\n"
        "import modules.evaluate.src.evaluation.reservation\n"
        "import modules.evaluate.src.training\n"
        "import modules.evaluate.src.predictors\n"
        "import modules.evaluate.src.scoring\n"
        "loaded = [m for m in sys.modules if 'dataset_sourcing' in m]\n"
        "print(loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"
