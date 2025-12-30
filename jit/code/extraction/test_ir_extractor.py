"""Test IR extraction with various function types."""
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir, extract_ir_pair

# Test 1: Simple array addition
@jax.jit
def add_function(a, b):
    """Simple elementwise addition."""
    return a + b

# Test 2: Reduction (dot product)
@jax.jit
def dot_product(a, b):
    """Dot product via reduction."""
    return jnp.sum(a * b)

# Test 3: Scalar + array operations (SAXPY)
@jax.jit
def saxpy(alpha, x, y):
    """SAXPY operation."""
    return alpha * x + y

# Test 4: Branching function
@jax.jit
def branch_function(a):
    """Conditional operation."""
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)

# Test 5: Loop function (using scan with fixed length)
@jax.jit
def loop_function(a):
    """Loop operation using scan."""
    def body_fn(carry, _):
        return carry + a, None
    result, _ = jax.lax.scan(body_fn, a, None, length=5)
    return result

# Test 6: Vector operations
@jax.jit
def vec_function(a, b):
    """Vector dot product."""
    return jnp.dot(a, b)


def run_tests():
    test_cases = [
        ("add_function", add_function, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("dot_product", dot_product, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("saxpy", saxpy, (2.0, jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
        ("branch_function", branch_function, (jnp.array([1.0, -2.0, 3.0]),)),
        ("loop_function", loop_function, (jnp.array([1.0, 2.0, 3.0]),)),
        ("vec_function", vec_function, (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))),
    ]
    
    results = []
    for name, func, args in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(func, *args)
            py_src, hlo_code = ir.python_source, ir.hlo_text
            
            print(f"Python source length: {len(py_src)} chars")
            print(f"HLO code length: {len(hlo_code)} chars")
            print(f"Has forward pass: {'func.func' in hlo_code}")
            print(f"Has backward pass: {'BACKWARD PASS' in hlo_code}")
            
            results.append((name, True, len(py_src), len(hlo_code)))
            
            # Print first part of Python source
            print(f"\nPython source:")
            print(py_src[:300])
            
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
