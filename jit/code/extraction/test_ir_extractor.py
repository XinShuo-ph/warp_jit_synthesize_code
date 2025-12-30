"""Test IR extraction with various function types."""
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir, extract_ir_pair


# Test 1: Simple array addition
def add_kernel(a, b):
    return a + b


# Test 2: Reduction (dot product)
def dot_product(a, b):
    return jnp.sum(a * b)


# Test 3: Scalar + array operations (SAXPY)
def saxpy(alpha, x, y):
    return alpha * x + y


# Test 4: Branching kernel (using jnp.where)
def branch_kernel(a):
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)


# Test 5: Loop kernel (using fori_loop)
def loop_kernel(a, n):
    def body_fn(i, total):
        return total + a
    return jax.lax.fori_loop(0, n, body_fn, jnp.zeros_like(a))


# Test 6: Vector operations (dot product per row)
def vec_kernel(a, b):
    return jnp.sum(a * b, axis=-1)


def run_tests():
    # Create sample inputs
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (100,))
    b = jax.random.normal(jax.random.PRNGKey(43), (100,))
    a_vec = jax.random.normal(key, (100, 3))
    b_vec = jax.random.normal(jax.random.PRNGKey(43), (100, 3))
    
    kernels = [
        ("add_kernel", add_kernel, (a, b)),
        ("dot_product", dot_product, (a, b)),
        ("saxpy", saxpy, (2.0, a, b)),
        ("branch_kernel", branch_kernel, (a,)),
        ("loop_kernel", loop_kernel, (a, 5)),
        ("vec_kernel", vec_kernel, (a_vec, b_vec)),
    ]
    
    results = []
    for name, func, sample_args in kernels:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(func, sample_args)
            hlo_code = ir.hlo_text
            jaxpr_code = ir.jaxpr_text
            
            print(f"Jaxpr length: {len(jaxpr_code)} chars")
            print(f"HLO code length: {len(hlo_code)} chars")
            print(f"Has forward pass: {'ENTRY' in hlo_code}")
            print(f"Has backward pass: {'BACKWARD' in hlo_code or 'GRADIENT' in hlo_code}")
            
            results.append((name, True, len(jaxpr_code), len(hlo_code)))
            
            # Print first part of Jaxpr
            print(f"\nJaxpr (first 500 chars):")
            print(jaxpr_code[:500])
            
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
    
    for name, success, jaxpr_len, hlo_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: jaxpr={jaxpr_len}, hlo={hlo_len}")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
