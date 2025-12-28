"""Basic warp kernel tests to verify environment setup."""
import sys
import warp as wp
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Run basic Warp kernels")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    return parser.parse_args()

# Example 1: Simple element-wise kernel
@wp.kernel
def add_arrays(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Example 2: Kernel with math operations
@wp.kernel  
def compute_sine_wave(output: wp.array(dtype=float), freq: float, amplitude: float):
    tid = wp.tid()
    x = float(tid) / 100.0
    output[tid] = amplitude * wp.sin(freq * x * 2.0 * 3.14159)

# Example 3: Vector operations kernel
@wp.kernel
def normalize_vectors(vectors: wp.array(dtype=wp.vec3), output: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = vectors[tid]
    length = wp.length(v)
    if length > 0.0:
        output[tid] = v / length
    else:
        output[tid] = wp.vec3(0.0, 0.0, 0.0)


def test_add_arrays(device):
    """Test 1: Simple array addition."""
    print(f"Running Test 1 on {device}...")
    n = 1024
    a = wp.array(np.ones(n, dtype=np.float32), dtype=float, device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 2.0, dtype=float, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    
    wp.launch(add_arrays, dim=n, inputs=[a, b, c], device=device)
    wp.synchronize()
    
    result = c.numpy()
    expected = 3.0
    assert np.allclose(result, expected), f"Expected {expected}, got {result[:5]}"
    print(f"Test 1 PASSED: Array addition (sum={result[0]})")
    return True


def test_sine_wave(device):
    """Test 2: Sine wave computation."""
    print(f"Running Test 2 on {device}...")
    n = 256
    output = wp.zeros(n, dtype=float, device=device)
    
    wp.launch(compute_sine_wave, dim=n, inputs=[output, 1.0, 2.0], device=device)
    wp.synchronize()
    
    result = output.numpy()
    # Verify sine wave properties
    assert result[0] < 0.1, f"Expected ~0 at start, got {result[0]}"
    assert np.max(np.abs(result)) <= 2.1, f"Amplitude should be <= 2.0"
    print(f"Test 2 PASSED: Sine wave (max={np.max(result):.3f}, min={np.min(result):.3f})")
    return True


def test_vector_normalize(device):
    """Test 3: Vector normalization."""
    print(f"Running Test 3 on {device}...")
    n = 100
    vectors_np = np.random.randn(n, 3).astype(np.float32)
    vectors = wp.array(vectors_np, dtype=wp.vec3, device=device)
    output = wp.zeros(n, dtype=wp.vec3, device=device)
    
    wp.launch(normalize_vectors, dim=n, inputs=[vectors, output], device=device)
    wp.synchronize()
    
    result = output.numpy()
    lengths = np.linalg.norm(result, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5), f"Expected unit vectors, got lengths {lengths[:5]}"
    print(f"Test 3 PASSED: Vector normalization (all vectors unit length)")
    return True


if __name__ == "__main__":
    args = get_args()
    wp.init()
    
    print("=" * 50)
    print(f"Running Warp Basic Kernel Tests on {args.device}")
    print("=" * 50)
    
    if args.device == "cuda" and not wp.get_cuda_device_count():
        print("WARNING: CUDA requested but not available. Exiting.")
        # Exit with error code 0 or 1? If verifying, maybe 1 to signal failure to run.
        # But if just checking availability, print warning.
        sys.exit(1)

    tests = [test_add_arrays, test_sine_wave, test_vector_normalize]
    passed = 0
    
    for test in tests:
        try:
            if test(args.device):
                passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}: {e}")
    
    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
