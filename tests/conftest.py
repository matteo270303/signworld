"""Fixtures shared by every test package."""

# Imported before numpy and torch: the package caps the BLAS thread count, which those
# libraries read as they load (see signworld/__init__.py).
import signworld  # noqa: F401  (imported for its side effect)

# ruff: isort: off
from pathlib import Path

import pytest
import torch

from signworld.data.acquisition.config import AcquisitionConfig, HttpConfig, RefusalPolicy

# Small matrices run faster on a few threads than on every core of a shared login node.
torch.set_num_threads(4)


@pytest.fixture
def config(tmp_path: Path) -> AcquisitionConfig:
    return AcquisitionConfig(
        data_root=tmp_path / "data",
        max_attempts=2,
        refusals=RefusalPolicy(max_consecutive=3, cooldown_s=10.0),
        http=HttpConfig(chunk_bytes=1024, retries=3, timeout_s=5.0, backoff_s=0.0),
    )
