import warp as wp

# Ensure extraction utilities are importable
import sys

sys.path.insert(0, "/workspace/jit/code/extraction")

from ir_extractor import extract_ir  # noqa: E402


@wp.kernel
def kernel_smoke(a: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] + 1.0


def test_cpu_codegen_smoke():
    wp.init()

    result = extract_ir(kernel_smoke, device="cpu", include_backward=False)
    assert result["python_source"]
    assert result["cpp_code"]
    assert result["forward_code"]
    assert result["metadata"]["device"] == "cpu"

