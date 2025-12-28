"""Pytest coverage for IR extraction on CPU/CUDA."""

import numpy as np
import pytest
import warp as wp

from ir_extractor import extract_ir, extract_kernel_functions
from code.extraction.offline_codegen import codegen_module_source


wp.init()


@wp.kernel
def kernel_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


@wp.kernel
def kernel_dot_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])


@wp.kernel
def kernel_mat_mul(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]


@wp.kernel
def kernel_clamp(arr: wp.array(dtype=float), min_val: float, max_val: float):
    tid = wp.tid()
    val = arr[tid]
    if val < min_val:
        arr[tid] = min_val
    elif val > max_val:
        arr[tid] = max_val


@wp.kernel
def kernel_sum_neighbors(input: wp.array(dtype=float), output: wp.array(dtype=float), width: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(-1, 2):
        idx = tid + i
        if idx >= 0 and idx < width:
            total = total + input[idx]
    output[tid] = total


@wp.kernel
def kernel_math_ops(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    y[tid] = wp.sin(val) * wp.cos(val) + wp.exp(-val * val)


@wp.kernel
def kernel_atomic_add(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])


KERNELS = [
    kernel_add,
    kernel_dot_product,
    kernel_mat_mul,
    kernel_clamp,
    kernel_sum_neighbors,
    kernel_math_ops,
    kernel_atomic_add,
]


def force_compile_kernels(device: str):
    """Force compilation by launching each kernel once on a given device."""
    n = 10

    a = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    b = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel_add, dim=n, inputs=[a, b, c], device=device)

    v1 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    v2 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    out_f = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel_dot_product, dim=n, inputs=[v1, v2, out_f], device=device)

    mats = wp.array(
        np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(n, axis=0),
        dtype=wp.mat33,
        device=device,
    )
    vecs = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    out_v = wp.zeros(n, dtype=wp.vec3, device=device)
    wp.launch(kernel_mat_mul, dim=n, inputs=[mats, vecs, out_v], device=device)

    arr = wp.array(np.random.randn(n).astype(np.float32), dtype=float, device=device)
    wp.launch(kernel_clamp, dim=n, inputs=[arr, -0.5, 0.5], device=device)

    inp = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    outp = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel_sum_neighbors, dim=n, inputs=[inp, outp, n], device=device)

    x = wp.array(np.linspace(0, 1, n).astype(np.float32), dtype=float, device=device)
    y = wp.zeros(n, dtype=float, device=device)
    wp.launch(kernel_math_ops, dim=n, inputs=[x, y], device=device)

    vals = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    res = wp.zeros(1, dtype=float, device=device)
    wp.launch(kernel_atomic_add, dim=n, inputs=[vals, res], device=device)

    wp.synchronize_device(device)


@pytest.mark.parametrize("device", ["cpu"])
def test_extract_ir_and_forward_function(device: str):
    force_compile_kernels(device)

    for kernel in KERNELS:
        ir_pair = extract_ir(kernel, device=device)
        assert ir_pair.kernel_name == kernel.key
        assert ir_pair.python_source
        assert ir_pair.cpp_ir

        funcs = extract_kernel_functions(ir_pair.cpp_ir, kernel.key, device=device)
        assert funcs["forward"], f"missing forward function for {kernel.key} on {device}"


@pytest.mark.cuda
def test_extract_ir_and_forward_function_cuda():
    if not wp.is_cuda_available():
        pytest.skip("CUDA device not available")

    device = "cuda"
    force_compile_kernels(device)

    for kernel in KERNELS:
        ir_pair = extract_ir(kernel, device=device)
        funcs = extract_kernel_functions(ir_pair.cpp_ir, kernel.key, device=device)
        assert funcs["forward"], f"missing forward function for {kernel.key} on {device}"


@pytest.mark.cuda_codegen
def test_offline_cuda_codegen_extract_forward_function():
    # This must work on CPU-only machines: generate CUDA code via internal codegen
    # without loading kernels onto a CUDA device.
    try:
        cg = codegen_module_source(kernel_add.module, target="cuda", enable_backward=True)
    except Exception as e:
        pytest.skip(f"CUDA codegen not available in this Warp build/environment: {e}")

    funcs = extract_kernel_functions(cg.source, kernel_add.key, device="cuda")
    assert funcs["forward"], "missing forward function in offline CUDA codegen output"
