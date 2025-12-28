#!/usr/bin/env python3
"""
GPU Execution Tests - REQUIRES NVIDIA GPU with CUDA.

Run this script on a machine with a GPU to verify the generated CUDA code
actually executes correctly.

Usage:
    python run_gpu_tests.py [--quick]
    
Options:
    --quick     Run only a few tests instead of full suite
"""
import sys
import json
import argparse
from pathlib import Path

# Add code directories
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))

import warp as wp
import numpy as np


def check_gpu_available():
    """Check if GPU is available."""
    wp.init()
    
    devices = wp.get_devices()
    cuda_devices = [d for d in devices if "cuda" in d.alias.lower()]
    
    if not cuda_devices:
        print("=" * 60)
        print("ERROR: No CUDA GPU found!")
        print("This test requires an NVIDIA GPU with CUDA support.")
        print("=" * 60)
        print("\nAvailable devices:")
        for d in devices:
            print(f"  - {d.alias}")
        return False
    
    print(f"Found GPU: {cuda_devices[0].alias}")
    return True


def test_simple_kernel_execution():
    """Test that a simple kernel runs on GPU."""
    @wp.kernel
    def add_kernel(a: wp.array(dtype=float), 
                   b: wp.array(dtype=float), 
                   c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]
    
    n = 1000
    a = wp.array(np.random.randn(n).astype(np.float32), device="cuda")
    b = wp.array(np.random.randn(n).astype(np.float32), device="cuda")
    c = wp.zeros(n, dtype=float, device="cuda")
    
    wp.launch(add_kernel, dim=n, inputs=[a, b, c], device="cuda")
    wp.synchronize()
    
    # Verify results
    a_np = a.numpy()
    b_np = b.numpy()
    c_np = c.numpy()
    
    expected = a_np + b_np
    assert np.allclose(c_np, expected, rtol=1e-5), "Add kernel results mismatch"
    
    print("✓ test_simple_kernel_execution passed")


def test_math_kernel_execution():
    """Test math functions on GPU."""
    @wp.kernel
    def math_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float)):
        tid = wp.tid()
        val = wp.sin(x[tid]) * wp.cos(x[tid])
        out[tid] = wp.exp(val * 0.1)
    
    n = 1000
    x = wp.array(np.random.randn(n).astype(np.float32), device="cuda")
    out = wp.zeros(n, dtype=float, device="cuda")
    
    wp.launch(math_kernel, dim=n, inputs=[x, out], device="cuda")
    wp.synchronize()
    
    # Verify results
    x_np = x.numpy()
    expected = np.exp(np.sin(x_np) * np.cos(x_np) * 0.1)
    out_np = out.numpy()
    
    assert np.allclose(out_np, expected, rtol=1e-4), "Math kernel results mismatch"
    
    print("✓ test_math_kernel_execution passed")


def test_vector_kernel_execution():
    """Test vector operations on GPU."""
    @wp.kernel
    def vec_kernel(pos: wp.array(dtype=wp.vec3),
                   vel: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        dt = 0.01
        pos[tid] = pos[tid] + vel[tid] * dt
    
    n = 100
    pos_data = np.random.randn(n, 3).astype(np.float32)
    vel_data = np.random.randn(n, 3).astype(np.float32)
    
    pos = wp.array(pos_data, dtype=wp.vec3, device="cuda")
    vel = wp.array(vel_data, dtype=wp.vec3, device="cuda")
    
    wp.launch(vec_kernel, dim=n, inputs=[pos, vel], device="cuda")
    wp.synchronize()
    
    expected = pos_data + vel_data * 0.01
    result = pos.numpy()
    
    assert np.allclose(result, expected, rtol=1e-5), "Vector kernel results mismatch"
    
    print("✓ test_vector_kernel_execution passed")


def test_atomic_kernel_execution():
    """Test atomic operations on GPU."""
    @wp.kernel
    def sum_kernel(values: wp.array(dtype=float),
                   result: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(result, 0, values[tid])
    
    n = 1000
    values = wp.array(np.ones(n, dtype=np.float32), device="cuda")
    result = wp.zeros(1, dtype=float, device="cuda")
    
    wp.launch(sum_kernel, dim=n, inputs=[values, result], device="cuda")
    wp.synchronize()
    
    total = result.numpy()[0]
    assert abs(total - n) < 1e-3, f"Atomic sum mismatch: expected {n}, got {total}"
    
    print("✓ test_atomic_kernel_execution passed")


def test_generated_kernel_execution():
    """Test dynamically generated kernel on GPU."""
    from generator import KernelGenerator
    from pipeline import compile_kernel_from_source
    
    gen = KernelGenerator(seed=42)
    spec = gen.generate("arithmetic")
    source = gen.to_python_source(spec)
    
    # Compile the kernel
    kernel = compile_kernel_from_source(source, spec.name)
    
    # Create test data
    n = 100
    a = wp.array(np.random.randn(n).astype(np.float32), device="cuda")
    b = wp.array(np.random.randn(n).astype(np.float32), device="cuda")
    out = wp.zeros(n, dtype=float, device="cuda")
    
    # Launch the kernel
    wp.launch(kernel, dim=n, inputs=[a, b, out], device="cuda")
    wp.synchronize()
    
    # Just verify it ran without error
    result = out.numpy()
    assert result.shape == (n,)
    assert not np.any(np.isnan(result)), "Generated kernel produced NaN"
    
    print("✓ test_generated_kernel_execution passed")


def test_backward_execution():
    """Test backward pass (gradient computation) on GPU."""
    @wp.kernel
    def square_kernel(x: wp.array(dtype=float), out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = x[tid] * x[tid]
    
    n = 100
    x = wp.array(np.random.randn(n).astype(np.float32), device="cuda", requires_grad=True)
    out = wp.zeros(n, dtype=float, device="cuda", requires_grad=True)
    
    tape = wp.Tape()
    with tape:
        wp.launch(square_kernel, dim=n, inputs=[x, out], device="cuda")
    
    # Set gradient at output
    out.grad = wp.array(np.ones(n, dtype=np.float32), device="cuda")
    
    # Backward pass
    tape.backward()
    
    # Gradient should be 2*x
    x_np = x.numpy()
    grad_np = x.grad.numpy()
    expected_grad = 2.0 * x_np
    
    assert np.allclose(grad_np, expected_grad, rtol=1e-4), "Gradient mismatch"
    
    print("✓ test_backward_execution passed")


def run_all_tests(quick=False):
    """Run all GPU tests."""
    print("=" * 60)
    print("GPU Execution Tests (REQUIRES CUDA GPU)")
    print("=" * 60)
    
    if not check_gpu_available():
        return False
    
    print()
    
    if quick:
        tests = [
            test_simple_kernel_execution,
            test_math_kernel_execution,
        ]
    else:
        tests = [
            test_simple_kernel_execution,
            test_math_kernel_execution,
            test_vector_kernel_execution,
            test_atomic_kernel_execution,
            test_generated_kernel_execution,
            test_backward_execution,
        ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPU execution tests")
    parser.add_argument("--quick", action="store_true", help="Run quick test subset")
    args = parser.parse_args()
    
    success = run_all_tests(quick=args.quick)
    sys.exit(0 if success else 1)
