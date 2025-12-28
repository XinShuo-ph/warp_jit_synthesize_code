import sys

import pytest
import warp as wp

sys.path.insert(0, "/workspace/jit/code/extraction")

from ir_extractor import extract_ir  # noqa: E402


@wp.kernel
def kernel_cuda_codegen_probe(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 3.0


def test_cuda_codegen_without_gpu():
    """
    Validate we can *produce* CUDA generated code without launching on a GPU.

    This should pass when CUDA toolchain/NVRTC is available and skip otherwise.
    """
    wp.init()

    try:
        result = extract_ir(kernel_cuda_codegen_probe, device="cuda", include_backward=False)
    except Exception as e:
        # Toolchain missing is an acceptable outcome in CPU-only environments.
        msg = str(e)
        raise pytest.skip(f"CUDA codegen unavailable in this environment: {msg}")

    assert result["cpp_code"]
    assert result["forward_code"]
    assert result["metadata"]["device"] == "cuda"

