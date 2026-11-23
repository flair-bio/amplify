import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("layout", ["split", "single"])
def test_native_training_export_and_independent_reload(tmp_path, layout):
    helper = Path(__file__).with_name("native_fixture.py")
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd()), "HF_HUB_OFFLINE": "1"}
    for phase in ("build", "train", "reload"):
        if phase == "reload":
            # The export must not depend on either original inputs or heads.
            shutil.rmtree(tmp_path / "student")
            shutil.rmtree(tmp_path / "data")
            (tmp_path / "output/alignment_heads.pt").unlink()
        result = subprocess.run(
            [sys.executable, str(helper), phase, str(tmp_path), layout],
            env=environment, capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, f"{phase}:\n{result.stdout}\n{result.stderr}"
        if phase == "train":
            assert '"step":' not in result.stdout
            assert "Completed" in result.stderr
