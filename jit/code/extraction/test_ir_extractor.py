"""Test IR extraction with various kernel types."""
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

import warp as wp
from ir_extractor import extract_ir, extract_ir_pair

wp.init()

# Test 1: Simple array addition
@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Test 2: Atomic operations (dot product)
@wp.kernel
def dot_product(a: wp.array(dtype=float), b: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, a[tid] * b[tid])

# Test 3: Scalar + array operations (SAXPY)
@wp.kernel
def saxpy(alpha: float, x: wp.array(dtype=float), y: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = alpha * x[tid] + y[tid]

# Test 4: Branching kernel
@wp.kernel
def branch_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val > 0.0:
        b[tid] = val * 2.0
    else:
        b[tid] = val * -1.0

# Test 5: Loop kernel
@wp.kernel
def loop_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), n: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(n):
        total = total + a[tid]
    b[tid] = total

# Test 6: Vector operations
@wp.kernel
def vec_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = wp.dot(a[tid], b[tid])


def run_tests():
    kernels = [
        ("add_kernel", add_kernel),
        ("dot_product", dot_product),
        ("saxpy", saxpy),
        ("branch_kernel", branch_kernel),
        ("loop_kernel", loop_kernel),
        ("vec_kernel", vec_kernel),
    ]
    
    results = []
    for name, kernel in kernels:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(kernel)
            py_src, cpp_code = ir.python_source, ir.cpp_code
            
            print(f"Python source length: {len(py_src)} chars")
            print(f"C++ code length: {len(cpp_code)} chars")
            print(f"Has forward pass: {'_forward' in cpp_code}")
            print(f"Has backward pass: {'_backward' in cpp_code}")
            
            results.append((name, True, len(py_src), len(cpp_code)))
            
            # Print first part of Python source
            print(f"\nPython source:")
            print(py_src[:300])
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append((name, False, 0, 0))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    success_count = sum(1 for _, success, _, _ in results if success)
    print(f"Passed: {success_count}/{len(results)}")
    
    for name, success, py_len, cpp_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: py={py_len}, cpp={cpp_len}")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
