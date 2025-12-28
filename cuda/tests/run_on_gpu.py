"""
GPU Execution Tests - Run CUDA kernels on actual GPU hardware.

WARNING: This test REQUIRES a CUDA-capable GPU and CUDA drivers installed.
It will fail on CPU-only systems.

This test:
1. Generates CUDA kernels
2. Compiles and executes them on GPU
3. Verifies correctness against expected outputs
4. Measures basic performance
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code/extraction"))
sys.path.insert(0, str(Path(__file__).parent.parent / "code/synthesis"))

import warp as wp


def check_gpu_available():
    """Check if CUDA GPU is available."""
    print("=" * 70)
    print("Checking GPU Availability")
    print("=" * 70)
    
    wp.init()
    
    devices = wp.get_devices()
    print(f"Available devices: {devices}")
    
    cuda_devices = [d for d in devices if 'cuda' in str(d)]
    
    if not cuda_devices:
        print("\n✗ No CUDA devices found!")
        print("This test requires a CUDA-capable GPU.")
        print("\nTroubleshooting:")
        print("  1. Check GPU: nvidia-smi")
        print("  2. Check CUDA drivers installed")
        print("  3. Check Warp installation: pip install --upgrade warp-lang")
        return False
    
    print(f"\n✓ Found {len(cuda_devices)} CUDA device(s)")
    for dev in cuda_devices:
        print(f"  - {dev}")
    
    return True


def test_simple_arithmetic():
    """Test simple arithmetic kernel on GPU."""
    print("\n" + "=" * 70)
    print("Test: Simple Arithmetic Kernel on GPU")
    print("=" * 70)
    
    # Define kernel
    @wp.kernel
    def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]
    
    # Create test data
    n = 1024
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    expected = a + b
    
    # Allocate GPU arrays
    a_gpu = wp.array(a, dtype=wp.float32, device="cuda")
    b_gpu = wp.array(b, dtype=wp.float32, device="cuda")
    c_gpu = wp.zeros(n, dtype=wp.float32, device="cuda")
    
    print(f"  Running kernel on {n} elements...")
    
    # Launch kernel
    wp.launch(add_kernel, dim=n, inputs=[a_gpu, b_gpu, c_gpu], device="cuda")
    wp.synchronize()
    
    # Copy result back
    result = c_gpu.numpy()
    
    # Verify
    max_error = np.max(np.abs(result - expected))
    print(f"  Max error: {max_error:.2e}")
    
    assert max_error < 1e-6, f"Error too large: {max_error}"
    print("  ✓ Results match expected values")
    
    return True


def test_vector_operations():
    """Test vector operations on GPU."""
    print("\n" + "=" * 70)
    print("Test: Vector Operations on GPU")
    print("=" * 70)
    
    @wp.kernel
    def dot_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = wp.dot(a[tid], b[tid])
    
    # Create test data
    n = 512
    a = np.random.rand(n, 3).astype(np.float32)
    b = np.random.rand(n, 3).astype(np.float32)
    expected = np.sum(a * b, axis=1)
    
    # Allocate GPU arrays
    a_gpu = wp.array(a, dtype=wp.vec3, device="cuda")
    b_gpu = wp.array(b, dtype=wp.vec3, device="cuda")
    c_gpu = wp.zeros(n, dtype=wp.float32, device="cuda")
    
    print(f"  Running vector kernel on {n} vec3...")
    
    # Launch kernel
    wp.launch(dot_kernel, dim=n, inputs=[a_gpu, b_gpu, c_gpu], device="cuda")
    wp.synchronize()
    
    # Verify
    result = c_gpu.numpy()
    max_error = np.max(np.abs(result - expected))
    print(f"  Max error: {max_error:.2e}")
    
    assert max_error < 1e-5, f"Error too large: {max_error}"
    print("  ✓ Vector operations correct")
    
    return True


def test_atomic_operations():
    """Test atomic operations on GPU."""
    print("\n" + "=" * 70)
    print("Test: Atomic Operations on GPU")
    print("=" * 70)
    
    @wp.kernel
    def sum_kernel(values: wp.array(dtype=float), result: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(result, 0, values[tid])
    
    # Create test data
    n = 1000
    values = np.ones(n, dtype=np.float32)
    expected_sum = float(n)
    
    # Allocate GPU arrays
    values_gpu = wp.array(values, dtype=wp.float32, device="cuda")
    result_gpu = wp.zeros(1, dtype=wp.float32, device="cuda")
    
    print(f"  Running atomic sum on {n} elements...")
    
    # Launch kernel
    wp.launch(sum_kernel, dim=n, inputs=[values_gpu, result_gpu], device="cuda")
    wp.synchronize()
    
    # Verify
    result = result_gpu.numpy()[0]
    error = abs(result - expected_sum)
    print(f"  Result: {result}, Expected: {expected_sum}, Error: {error:.2e}")
    
    assert error < 1e-4, f"Error too large: {error}"
    print("  ✓ Atomic operations correct")
    
    return True


def test_performance_comparison():
    """Compare CPU vs GPU performance."""
    print("\n" + "=" * 70)
    print("Test: CPU vs GPU Performance")
    print("=" * 70)
    
    @wp.kernel
    def compute_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
        tid = wp.tid()
        val = a[tid]
        for i in range(100):
            val = wp.sin(val) + wp.cos(val)
        b[tid] = val
    
    n = 10000
    a_data = np.random.rand(n).astype(np.float32)
    
    # CPU timing
    print(f"\n  Testing on {n} elements with 100 math ops each...")
    print("  CPU:")
    
    a_cpu = wp.array(a_data, dtype=wp.float32, device="cpu")
    b_cpu = wp.zeros(n, dtype=wp.float32, device="cpu")
    
    import time
    wp.synchronize()
    start = time.time()
    wp.launch(compute_kernel, dim=n, inputs=[a_cpu, b_cpu], device="cpu")
    wp.synchronize()
    cpu_time = time.time() - start
    print(f"    Time: {cpu_time*1000:.2f} ms")
    
    # GPU timing
    print("  GPU:")
    a_gpu = wp.array(a_data, dtype=wp.float32, device="cuda")
    b_gpu = wp.zeros(n, dtype=wp.float32, device="cuda")
    
    # Warmup
    wp.launch(compute_kernel, dim=n, inputs=[a_gpu, b_gpu], device="cuda")
    wp.synchronize()
    
    start = time.time()
    wp.launch(compute_kernel, dim=n, inputs=[a_gpu, b_gpu], device="cuda")
    wp.synchronize()
    gpu_time = time.time() - start
    print(f"    Time: {gpu_time*1000:.2f} ms")
    
    speedup = cpu_time / gpu_time
    print(f"\n  Speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("  ✓ GPU faster than CPU")
    else:
        print("  ⚠ GPU not faster (may be expected for small workload)")
    
    return True


def main():
    """Run all GPU tests."""
    print("\n" + "=" * 70)
    print("CUDA GPU Execution Tests")
    print("=" * 70)
    
    # Check GPU
    if not check_gpu_available():
        print("\n✗ CANNOT RUN GPU TESTS - No CUDA GPU available")
        return 1
    
    try:
        # Run tests
        test_simple_arithmetic()
        test_vector_operations()
        test_atomic_operations()
        test_performance_comparison()
        
        print("\n" + "=" * 70)
        print("✓ ALL GPU TESTS PASSED")
        print("=" * 70)
        print("\nYour GPU is working correctly with Warp CUDA kernels!")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ GPU TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
