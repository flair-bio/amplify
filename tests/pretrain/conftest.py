"""Fixtures shared by the pretrain tests."""

import pytest


@pytest.fixture
def gloo_group(tmp_path):
    """A single-process gloo group, so a real DDP can be constructed on CPU."""
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    yield
    dist.destroy_process_group()
