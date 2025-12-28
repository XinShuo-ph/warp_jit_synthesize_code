"""
CUDA smoke test for Warp codegen + IR extraction.

This script is safe to run on CPU-only machines: it will exit 0 with a clear message.
Run this on a CUDA machine to validate CUDA codegen end-to-end.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from pathlib import Path
import sys

# Allow importing extraction utilities when run from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
from ir_extractor import extract_ir


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] + b[tid]


def main() -> int:
    wp.init()

    if not wp.is_cuda_available():
        print("CUDA not available (Warp). Skipping CUDA smoke test.")
        return 0

    n = 16
    device = "cuda"

    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.array(np.arange(n, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)

    wp.launch(add_kernel, dim=n, inputs=[a, b, out], device=device)
    wp.synchronize_device(device)

    ir = extract_ir(add_kernel, device=device, include_backward=False)
    assert ir["forward_code"], "Expected CUDA forward kernel code to be present"

    print("CUDA smoke test passed (forward code extracted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

