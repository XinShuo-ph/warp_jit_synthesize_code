from __future__ import annotations

import sys
from pathlib import Path

import warp as wp


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "jit" / "code" / "extraction"))

from ir_extractor import extract_ir  # noqa: E402


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


def test_cpu_ir_extraction_smoke():
    wp.init()
    ir = extract_ir(add_kernel, device="cpu", include_backward=False)
    assert ir["python_source"]
    assert ir["cpp_code"]
    assert ir["forward_code"]

