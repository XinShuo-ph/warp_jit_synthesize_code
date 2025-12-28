"""
Comprehensive CUDA kernel test suite.
Run this on a machine with GPU to validate CUDA code generation.
"""
import warp as wp
import numpy as np
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def check_cuda_available():
    """Check if CUDA is available."""
    wp.init()
    cuda_devices = [d for d in wp.get_devices() if d.is_cuda]
    
    if not cuda_devices:
        print(f"{Colors.RED}✗ No CUDA devices found{Colors.END}")
        print(f"{Colors.YELLOW}Available devices: {wp.get_devices()}{Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✓ CUDA devices found: {len(cuda_devices)}{Colors.END}")
    for device in cuda_devices:
        print(f"  - {device}")
    return True


def test_arithmetic_kernel():
    """Test simple arithmetic kernel."""
    print(f"\n{Colors.BOLD}Test: Arithmetic Kernel{Colors.END}")
    
    @wp.kernel
    def add_mul(a: wp.array(dtype=float),
                b: wp.array(dtype=float),
                c: wp.array(dtype=float),
                out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = (a[tid] + b[tid]) * c[tid]
    
    n = 1024
    a = wp.array(np.ones(n, dtype=np.float32), device="cuda")
    b = wp.array(np.ones(n, dtype=np.float32) * 2.0, device="cuda")
    c = wp.array(np.ones(n, dtype=np.float32) * 3.0, device="cuda")
    out = wp.array(np.zeros(n, dtype=np.float32), device="cuda")
    
    wp.launch(add_mul, dim=n, inputs=[a, b, c, out], device="cuda")
    wp.synchronize()
    
    result = out.numpy()
    expected = (1.0 + 2.0) * 3.0
    
    if np.allclose(result, expected):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Result matches expected ({expected})")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Expected {expected}, got {result[0]}")
        return False


def test_vector_kernel():
    """Test vector operations kernel."""
    print(f"\n{Colors.BOLD}Test: Vector Kernel{Colors.END}")
    
    @wp.kernel
    def vec_dot(a: wp.array(dtype=wp.vec3),
                b: wp.array(dtype=wp.vec3),
                out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = wp.dot(a[tid], b[tid])
    
    n = 1024
    a_data = np.array([[1.0, 0.0, 0.0]] * n, dtype=np.float32)
    b_data = np.array([[1.0, 1.0, 0.0]] * n, dtype=np.float32)
    
    a = wp.array(a_data, dtype=wp.vec3, device="cuda")
    b = wp.array(b_data, dtype=wp.vec3, device="cuda")
    out = wp.array(np.zeros(n, dtype=np.float32), device="cuda")
    
    wp.launch(vec_dot, dim=n, inputs=[a, b, out], device="cuda")
    wp.synchronize()
    
    result = out.numpy()
    expected = 1.0  # dot([1,0,0], [1,1,0]) = 1
    
    if np.allclose(result, expected):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Vector dot product correct")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Expected {expected}, got {result[0]}")
        return False


def test_matrix_kernel():
    """Test matrix operations kernel."""
    print(f"\n{Colors.BOLD}Test: Matrix Kernel{Colors.END}")
    
    @wp.kernel
    def mat_transpose(m: wp.array(dtype=wp.mat22),
                      out: wp.array(dtype=wp.mat22)):
        tid = wp.tid()
        out[tid] = wp.transpose(m[tid])
    
    n = 1024
    # Create matrix [[1, 2], [3, 4]]
    m_data = np.array([[[1.0, 2.0], [3.0, 4.0]]] * n, dtype=np.float32)
    
    m = wp.array(m_data, dtype=wp.mat22, device="cuda")
    out = wp.array(np.zeros((n, 2, 2), dtype=np.float32), dtype=wp.mat22, device="cuda")
    
    wp.launch(mat_transpose, dim=n, inputs=[m, out], device="cuda")
    wp.synchronize()
    
    result = out.numpy()
    # Transpose: [[1, 3], [2, 4]]
    expected = np.array([[1.0, 3.0], [2.0, 4.0]])
    
    if np.allclose(result[0], expected):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Matrix transpose correct")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Transpose mismatch")
        print(f"    Expected:\n{expected}")
        print(f"    Got:\n{result[0]}")
        return False


def test_control_flow_kernel():
    """Test conditional logic kernel."""
    print(f"\n{Colors.BOLD}Test: Control Flow Kernel{Colors.END}")
    
    @wp.kernel
    def clamp_kernel(a: wp.array(dtype=float),
                     out: wp.array(dtype=float)):
        tid = wp.tid()
        val = a[tid]
        if val < 0.0:
            out[tid] = 0.0
        elif val > 1.0:
            out[tid] = 1.0
        else:
            out[tid] = val
    
    n = 5
    a = wp.array(np.array([-1.0, 0.5, 0.0, 1.5, 1.0], dtype=np.float32), device="cuda")
    out = wp.array(np.zeros(n, dtype=np.float32), device="cuda")
    
    wp.launch(clamp_kernel, dim=n, inputs=[a, out], device="cuda")
    wp.synchronize()
    
    result = out.numpy()
    expected = np.array([0.0, 0.5, 0.0, 1.0, 1.0])
    
    if np.allclose(result, expected):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Conditional logic works")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result}")
        return False


def test_math_kernel():
    """Test math functions kernel."""
    print(f"\n{Colors.BOLD}Test: Math Functions Kernel{Colors.END}")
    
    @wp.kernel
    def math_ops(a: wp.array(dtype=float),
                 out: wp.array(dtype=float)):
        tid = wp.tid()
        val = a[tid]
        out[tid] = wp.sin(val) + wp.cos(val)
    
    n = 1024
    a = wp.array(np.zeros(n, dtype=np.float32), device="cuda")  # sin(0) + cos(0) = 1
    out = wp.array(np.zeros(n, dtype=np.float32), device="cuda")
    
    wp.launch(math_ops, dim=n, inputs=[a, out], device="cuda")
    wp.synchronize()
    
    result = out.numpy()
    expected = 1.0
    
    if np.allclose(result, expected, rtol=1e-5):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Math functions work")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Expected {expected}, got {result[0]}")
        return False


def test_atomic_kernel():
    """Test atomic operations kernel."""
    print(f"\n{Colors.BOLD}Test: Atomic Operations Kernel{Colors.END}")
    
    @wp.kernel
    def atomic_sum(values: wp.array(dtype=float),
                   result: wp.array(dtype=float)):
        tid = wp.tid()
        wp.atomic_add(result, 0, values[tid])
    
    n = 1000
    values = wp.array(np.ones(n, dtype=np.float32), device="cuda")
    result = wp.array(np.zeros(1, dtype=np.float32), device="cuda")
    
    wp.launch(atomic_sum, dim=n, inputs=[values, result], device="cuda")
    wp.synchronize()
    
    total = result.numpy()[0]
    expected = float(n)
    
    if np.isclose(total, expected):
        print(f"  {Colors.GREEN}✓ PASS{Colors.END}: Atomic add works ({total} == {expected})")
        return True
    else:
        print(f"  {Colors.RED}✗ FAIL{Colors.END}: Expected {expected}, got {total}")
        return False


def run_all_tests():
    """Run all test cases."""
    print("=" * 70)
    print(f"{Colors.BOLD}CUDA Kernel Test Suite{Colors.END}")
    print("=" * 70)
    
    # Check CUDA
    if not check_cuda_available():
        print(f"\n{Colors.RED}Cannot run tests without CUDA{Colors.END}")
        print(f"{Colors.YELLOW}This test suite requires a GPU with CUDA support{Colors.END}")
        return False
    
    # Run tests
    tests = [
        ("Arithmetic", test_arithmetic_kernel),
        ("Vector", test_vector_kernel),
        ("Matrix", test_matrix_kernel),
        ("Control Flow", test_control_flow_kernel),
        ("Math Functions", test_math_kernel),
        ("Atomic Operations", test_atomic_kernel),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"  {Colors.RED}✗ ERROR{Colors.END}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print(f"{Colors.BOLD}Test Summary{Colors.END}")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for name, result in results.items():
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"  {name:20s}: {status}")
    
    print(f"\n  {Colors.BOLD}Total: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.END}")
        return True
    else:
        print(f"\n  {Colors.YELLOW}Some tests failed. Check output above.{Colors.END}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
