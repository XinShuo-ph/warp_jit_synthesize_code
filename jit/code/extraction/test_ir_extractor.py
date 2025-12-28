"""Test cases for IR extractor with 5+ diverse kernel examples."""
import sys
sys.path.insert(0, "/workspace/jit/code/extraction")

import warp as wp
import numpy as np
from ir_extractor import extract_ir, extract_python_ir_pair

wp.init()

# ============================================================
# Test Kernel 1: Simple arithmetic
# ============================================================
@wp.kernel
def kernel_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# ============================================================
# Test Kernel 2: Vector operations
# ============================================================
@wp.kernel
def kernel_dot_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])

# ============================================================
# Test Kernel 3: Matrix operations
# ============================================================
@wp.kernel
def kernel_mat_mul(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]

# ============================================================
# Test Kernel 4: Control flow (if statement)
# ============================================================
@wp.kernel
def kernel_clamp(arr: wp.array(dtype=float), min_val: float, max_val: float):
    tid = wp.tid()
    val = arr[tid]
    if val < min_val:
        arr[tid] = min_val
    elif val > max_val:
        arr[tid] = max_val

# ============================================================
# Test Kernel 5: Loop (for statement)
# ============================================================
@wp.kernel
def kernel_sum_neighbors(input: wp.array(dtype=float), output: wp.array(dtype=float), width: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(-1, 2):
        idx = tid + i
        if idx >= 0 and idx < width:
            total = total + input[idx]
    output[tid] = total

# ============================================================
# Test Kernel 6: Built-in math functions
# ============================================================
@wp.kernel
def kernel_math_ops(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    y[tid] = wp.sin(val) * wp.cos(val) + wp.exp(-val * val)

# ============================================================
# Test Kernel 7: Atomic operations
# ============================================================
@wp.kernel
def kernel_atomic_add(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])


def force_compile_kernels():
    """Force compilation by launching each kernel once."""
    n = 10
    
    # Kernel 1
    a = wp.array(np.ones(n, dtype=np.float32))
    b = wp.array(np.ones(n, dtype=np.float32))
    c = wp.zeros(n, dtype=float)
    wp.launch(kernel_add, dim=n, inputs=[a, b, c])
    
    # Kernel 2
    v1 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    v2 = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    out_f = wp.zeros(n, dtype=float)
    wp.launch(kernel_dot_product, dim=n, inputs=[v1, v2, out_f])
    
    # Kernel 3
    mats = wp.array(np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(n, axis=0), dtype=wp.mat33)
    vecs = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3)
    out_v = wp.zeros(n, dtype=wp.vec3)
    wp.launch(kernel_mat_mul, dim=n, inputs=[mats, vecs, out_v])
    
    # Kernel 4
    arr = wp.array(np.random.randn(n).astype(np.float32))
    wp.launch(kernel_clamp, dim=n, inputs=[arr, -0.5, 0.5])
    
    # Kernel 5
    inp = wp.array(np.ones(n, dtype=np.float32))
    outp = wp.zeros(n, dtype=float)
    wp.launch(kernel_sum_neighbors, dim=n, inputs=[inp, outp, n])
    
    # Kernel 6
    x = wp.array(np.linspace(0, 1, n).astype(np.float32))
    y = wp.zeros(n, dtype=float)
    wp.launch(kernel_math_ops, dim=n, inputs=[x, y])
    
    # Kernel 7
    vals = wp.array(np.ones(n, dtype=np.float32))
    res = wp.zeros(1, dtype=float)
    wp.launch(kernel_atomic_add, dim=n, inputs=[vals, res])
    
    wp.synchronize()


def _run_ir_extraction_checks() -> bool:
    """Run the IR extraction checks (script-friendly)."""
    print("=" * 70)
    print("Testing IR Extraction")
    print("=" * 70)
    
    kernels = [
        ("Simple Arithmetic", kernel_add),
        ("Vector Operations", kernel_dot_product),
        ("Matrix Operations", kernel_mat_mul),
        ("Control Flow (if)", kernel_clamp),
        ("Loop (for)", kernel_sum_neighbors),
        ("Math Functions", kernel_math_ops),
        ("Atomic Operations", kernel_atomic_add),
    ]
    
    passed = 0
    for name, kernel in kernels:
        print(f"\n--- {name}: {kernel.key} ---")
        try:
            result = extract_ir(kernel)
            
            # Validate result
            assert result["python_source"], "Missing python_source"
            assert result["cpp_code"], "Missing cpp_code"
            assert result["forward_code"], "Missing forward_code"
            assert result["kernel_name"] == kernel.key, "Kernel name mismatch"
            
            # Print summary
            print(f"  Python source: {len(result['python_source'])} chars")
            print(f"  C++ code: {len(result['cpp_code'])} chars")
            print(f"  Forward func: {len(result['forward_code'])} chars")
            if result["backward_code"]:
                print(f"  Backward func: {len(result['backward_code'])} chars")
            print(f"  Args: {result['metadata']['arg_names']}")
            print(f"  PASSED ✓")
            passed += 1
            
        except Exception as e:
            print(f"  FAILED: {e}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(kernels)} kernels extracted successfully")
    print("=" * 70)
    
    return passed == len(kernels)


def test_ir_extraction():
    """pytest entrypoint: fails if any kernel extraction fails."""
    assert _run_ir_extraction_checks()


def show_sample_pair():
    """Show a sample Python→IR pair."""
    print("\n" + "=" * 70)
    print("Sample Python → IR Pair")
    print("=" * 70)
    
    result = extract_ir(kernel_add)
    
    print("\n--- Python Source ---")
    print(result["python_source"])
    
    print("\n--- Generated C++ Forward Kernel ---")
    print(result["forward_code"])


if __name__ == "__main__":
    force_compile_kernels()
    success = _run_ir_extraction_checks()
    if success:
        show_sample_pair()
    sys.exit(0 if success else 1)
