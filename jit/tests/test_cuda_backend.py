import pytest

import warp as wp

from jit.code.extraction.ir_extractor import extract_ir


wp.init()


@wp.kernel
def kernel_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


def test_extract_ir_cpu_smoke():
    result = extract_ir(kernel_add, device="cpu", include_backward=False)
    assert result["python_source"]
    assert result["cpp_code"]
    assert result["forward_code"]
    assert result["metadata"]["device"] == "cpu"
    assert result["metadata"]["code_ext"] == ".cpp"


def test_extract_ir_cuda_smoke():
    aliases = [d if isinstance(d, str) else getattr(d, "alias", str(d)) for d in wp.get_devices()]
    if "cuda" not in aliases:
        pytest.skip("CUDA not available in this environment")

    result = extract_ir(kernel_add, device="cuda", include_backward=False)
    assert result["python_source"]
    assert result["cpp_code"]
    assert result["forward_code"]
    assert result["metadata"]["device"] == "cuda"
    assert result["metadata"]["code_ext"] == ".cu"


def test_extract_ir_cuda_codegen_only_without_device():
    # Even if no CUDA device/driver exists, Warp can still codegen CUDA source.
    result = extract_ir(kernel_add, device="cuda", include_backward=False, require_device=False)
    assert result["cpp_code"]
    assert result["forward_code"]
    assert result["metadata"]["device"] == "cuda"
    assert result["metadata"]["code_ext"] == ".cu"
