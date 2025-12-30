"""Test IR extraction with various kernel types."""
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

import jax
import jax.numpy as jnp
import numpy as np
from ir_extractor import extract_ir, extract_ir_pair

# Test 1: Simple array addition
def add_kernel(a, b):
    """Elementwise addition"""
    return a + b

# Test 2: Reduction operations (dot product)
def dot_product(a, b):
    """Dot product via reduction"""
    return jnp.sum(a * b)

# Test 3: Scalar + array operations (SAXPY)
def saxpy(alpha, x, y):
    """SAXPY: alpha * x + y"""
    return alpha * x + y

# Test 4: Branching kernel
def branch_kernel(a):
    """Conditional operation"""
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)

# Test 5: Loop kernel
def loop_kernel(a, n):
    """Loop accumulation"""
    def body_fn(i, total):
        return total + a
    return jax.lax.fori_loop(0, n, body_fn, jnp.zeros_like(a))

# Test 6: Vector operations
def vec_kernel(a, b):
    """Vector dot product"""
    return jnp.sum(a * b, axis=-1)


def run_tests():
    kernels = [
        ("add_kernel", add_kernel, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("dot_product", dot_product, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("saxpy", saxpy, (2.0, jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("branch_kernel", branch_kernel, (jnp.array([-1.0, 0.5, 2.0]),)),
        ("loop_kernel", loop_kernel, (jnp.array([1.0, 2.0, 3.0]), 5)),
        ("vec_kernel", vec_kernel, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
    ]
    
    results = []
    for name, kernel, example_inputs in kernels:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(kernel, example_inputs, name)
            py_src, hlo_code = ir.python_source, ir.hlo_text
            
            print(f"Python source length: {len(py_src)} chars")
            print(f"HLO code length: {len(hlo_code)} chars")
            print(f"Has optimized HLO: {ir.optimized_hlo is not None}")
            print(f"Has MHLO: {ir.mhlo_text is not None}")
            
            results.append((name, True, len(py_src), len(hlo_code)))
            
            # Print first part of Python source
            print(f"\nPython source:")
            print(py_src[:300])
            
            # Print first part of HLO
            print(f"\nHLO (first 300 chars):")
            print(hlo_code[:300])
            
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, 0, 0))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    
    success_count = sum(1 for _, success, _, _ in results if success)
    print(f"Passed: {success_count}/{len(results)}")
    
    for name, success, py_len, hlo_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: py={py_len}, hlo={hlo_len}")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
