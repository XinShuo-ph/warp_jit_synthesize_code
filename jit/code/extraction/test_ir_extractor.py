"""Test IR extraction with various JAX kernel types."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jax.numpy as jnp
from jax import lax

from ir_extractor import extract_ir


# Test 1: Simple array addition
def add_kernel(a, b):
    return a + b


# Test 2: Reduction (dot product)
def dot_product(a, b):
    return jnp.sum(a * b)


# Test 3: Scalar + array operations (SAXPY)
def saxpy(alpha, x, y):
    return alpha * x + y


# Test 4: Branching (vectorized)
def branch_kernel(a):
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)


# Test 5: Loop kernel (we just test compilation/extraction here)
def loop_kernel(a, n):
    def body(i, acc):
        return acc + a

    init = jnp.zeros_like(a)
    return lax.fori_loop(0, n, body, init)


# Test 6: Vector operations (N, 3) -> (N,)
def vec_kernel(a, b):
    return jnp.sum(a * b, axis=-1)


def run_tests():
    n = 8
    kernels = [
        ("add_kernel", add_kernel, (jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32))),
        ("dot_product", dot_product, (jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32))),
        ("saxpy", saxpy, (2.0, jnp.arange(n, dtype=jnp.float32), jnp.arange(n, dtype=jnp.float32))),
        ("branch_kernel", branch_kernel, (jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32),)),
        ("loop_kernel", loop_kernel, (jnp.arange(n, dtype=jnp.float32), 3)),
        ("vec_kernel", vec_kernel, (jnp.ones((n, 3), dtype=jnp.float32), jnp.ones((n, 3), dtype=jnp.float32))),
    ]
    
    results = []
    for name, kernel, example_args in kernels:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(kernel, example_args=example_args)
            py_src, cpp_code = ir.python_source, ir.cpp_code
            
            print(f"Python source length: {len(py_src)} chars")
            print(f"C++ code length: {len(cpp_code)} chars")
            print(f"Has forward section: {'### FORWARD' in cpp_code}")
            print(f"Has backward section: {'### BACKWARD' in cpp_code}")
            
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
