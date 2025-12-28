from __future__ import annotations

import sys
from pathlib import Path

import pytest
import warp as wp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jit" / "code" / "synthesis"))

from generator import KernelSpec  # noqa: E402
from pipeline import synthesize_pair  # noqa: E402


def test_pipeline_cuda_unavailable_raises():
    wp.init()
    if wp.is_cuda_available():
        pytest.skip("CUDA is available; this test is for CPU-only machines.")

    spec = KernelSpec(
        name="dummy",
        category="arithmetic",
        source="@wp.kernel\ndef dummy(a: wp.array(dtype=float)):\n    tid = wp.tid()\n    a[tid] = a[tid]\n",
        arg_types={"a": "wp.array(dtype=float)"},
        description="Dummy kernel",
        metadata={},
    )

    # synthesize_pair is deliberately fail-soft (returns None on error)
    assert synthesize_pair(spec, device="cuda") is None

