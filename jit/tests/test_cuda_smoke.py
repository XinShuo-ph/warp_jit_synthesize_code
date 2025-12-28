from __future__ import annotations

import sys
from pathlib import Path

import warp as wp
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jit" / "code" / "examples"))

import smoke_cuda  # noqa: E402


@pytest.mark.cuda
def test_smoke_cuda_script():
    wp.init()

    if not wp.is_cuda_available():
        pytest.skip("CUDA not available (Warp).")

    assert smoke_cuda.main() == 0

