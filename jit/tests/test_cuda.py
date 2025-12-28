#!/usr/bin/env python3
"""
CUDA Backend Test Suite

Run this on a machine with CUDA-capable GPU to verify CUDA code generation works.

Usage:
    python3 tests/test_cuda.py
"""
import sys
import json
from pathlib import Path

# Add code directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "extraction"))

import warp as wp


def check_cuda_available():
    """Check if CUDA is available."""
    wp.init()
    if not wp.is_cuda_available():
        print("=" * 60)
        print("CUDA NOT AVAILABLE")
        print("=" * 60)
        print("This test requires a CUDA-capable GPU.")
        print("Please run on a machine with NVIDIA GPU and CUDA drivers.")
        print()
        print("To check CUDA setup:")
        print("  nvidia-smi")
        print("  python3 -c \"import warp as wp; wp.init(); print(wp.is_cuda_available())\"")
        return False
    
    print("=" * 60)
    print("CUDA AVAILABLE")
    print("=" * 60)
    print(f"Warp version: {wp.__version__}")
    devices = wp.get_cuda_devices()
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev}")
    print()
    return True


def test_simple_kernel_cuda():
    """Test simple kernel execution on CUDA."""
    print("Test: Simple kernel on CUDA")
    
    @wp.kernel
    def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]
    
    n = 1024
    a = wp.array([1.0] * n, dtype=float, device="cuda")
    b = wp.array([2.0] * n, dtype=float, device="cuda")
    c = wp.zeros(n, dtype=float, device="cuda")
    
    wp.launch(add_kernel, dim=n, inputs=[a, b, c], device="cuda")
    wp.synchronize()
    
    c_host = c.numpy()
    expected = 3.0
    if all(abs(v - expected) < 1e-6 for v in c_host):
        print("  PASS: Simple add kernel")
        return True
    else:
        print(f"  FAIL: Expected {expected}, got {c_host[:5]}...")
        return False


def test_atomic_kernel_cuda():
    """Test atomic operations on CUDA."""
    print("Test: Atomic operations on CUDA")
    
    @wp.kernel
    def atomic_sum(values: wp.array(dtype=float), result: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(result, 0, values[tid])
    
    n = 1000
    values = wp.array([1.0] * n, dtype=float, device="cuda")
    result = wp.zeros(1, dtype=float, device="cuda")
    
    wp.launch(atomic_sum, dim=n, inputs=[values, result], device="cuda")
    wp.synchronize()
    
    result_host = result.numpy()[0]
    expected = float(n)
    if abs(result_host - expected) < 1e-3:
        print("  PASS: Atomic sum")
        return True
    else:
        print(f"  FAIL: Expected {expected}, got {result_host}")
        return False


def test_vector_kernel_cuda():
    """Test vector operations on CUDA."""
    print("Test: Vector operations on CUDA")
    
    @wp.kernel
    def dot_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = wp.dot(a[tid], b[tid])
    
    n = 100
    a = wp.array([wp.vec3(1.0, 0.0, 0.0)] * n, dtype=wp.vec3, device="cuda")
    b = wp.array([wp.vec3(1.0, 0.0, 0.0)] * n, dtype=wp.vec3, device="cuda")
    out = wp.zeros(n, dtype=float, device="cuda")
    
    wp.launch(dot_kernel, dim=n, inputs=[a, b, out], device="cuda")
    wp.synchronize()
    
    out_host = out.numpy()
    expected = 1.0
    if all(abs(v - expected) < 1e-6 for v in out_host):
        print("  PASS: Vector dot product")
        return True
    else:
        print(f"  FAIL: Expected {expected}, got {out_host[:5]}...")
        return False


def test_matrix_kernel_cuda():
    """Test matrix operations on CUDA."""
    print("Test: Matrix operations on CUDA")
    
    @wp.kernel
    def mat_mul_kernel(a: wp.array(dtype=wp.mat33), b: wp.array(dtype=wp.mat33), c: wp.array(dtype=wp.mat33)):
        tid = wp.tid()
        c[tid] = a[tid] * b[tid]
    
    n = 100
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    a = wp.array([identity] * n, dtype=wp.mat33, device="cuda")
    b = wp.array([identity] * n, dtype=wp.mat33, device="cuda")
    c = wp.zeros(n, dtype=wp.mat33, device="cuda")
    
    wp.launch(mat_mul_kernel, dim=n, inputs=[a, b, c], device="cuda")
    wp.synchronize()
    
    c_host = c.numpy()
    # Check first element is identity
    if abs(c_host[0][0][0] - 1.0) < 1e-6:
        print("  PASS: Matrix multiply")
        return True
    else:
        print(f"  FAIL: Matrix multiply gave unexpected result")
        return False


def test_control_flow_cuda():
    """Test control flow on CUDA."""
    print("Test: Control flow on CUDA")
    
    @wp.kernel
    def clamp_kernel(a: wp.array(dtype=float), lo: float, hi: float, out: wp.array(dtype=float)):
        tid = wp.tid()
        val = a[tid]
        if val < lo:
            out[tid] = lo
        elif val > hi:
            out[tid] = hi
        else:
            out[tid] = val
    
    n = 100
    a = wp.array([float(i - 50) for i in range(n)], dtype=float, device="cuda")
    out = wp.zeros(n, dtype=float, device="cuda")
    
    wp.launch(clamp_kernel, dim=n, inputs=[a, -10.0, 10.0, out], device="cuda")
    wp.synchronize()
    
    out_host = out.numpy()
    if min(out_host) >= -10.0 and max(out_host) <= 10.0:
        print("  PASS: Control flow clamp")
        return True
    else:
        print(f"  FAIL: Values not clamped correctly")
        return False


def test_ir_generation_cuda():
    """Test IR extraction for CUDA."""
    print("Test: IR extraction for CUDA")
    
    from ir_extractor import extract_ir
    
    @wp.kernel
    def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
        tid = wp.tid()
        b[tid] = a[tid] * 2.0
    
    try:
        result = extract_ir(test_kernel, device="cuda")
        
        if result["forward_code"] and "_cuda_kernel_forward" in result["forward_code"]:
            print("  PASS: CUDA IR extraction")
            return True
        else:
            print("  FAIL: CUDA forward code not found")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_pipeline_cuda():
    """Test synthesis pipeline with CUDA."""
    print("Test: Synthesis pipeline for CUDA")
    
    from pipeline import synthesize_pair
    from generator import generate_kernel
    
    try:
        spec = generate_kernel("arithmetic", seed=123)
        pair = synthesize_pair(spec, device="cuda")
        
        if pair and "_cuda_kernel_forward" in pair["cpp_forward"]:
            print("  PASS: Pipeline generates CUDA code")
            return True
        else:
            print("  FAIL: Pipeline did not generate CUDA code")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def run_all_tests():
    """Run all CUDA tests."""
    if not check_cuda_available():
        return False
    
    tests = [
        test_simple_kernel_cuda,
        test_atomic_kernel_cuda,
        test_vector_kernel_cuda,
        test_matrix_kernel_cuda,
        test_control_flow_cuda,
        test_ir_generation_cuda,
        test_pipeline_cuda,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((test.__name__, False))
        print()
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
