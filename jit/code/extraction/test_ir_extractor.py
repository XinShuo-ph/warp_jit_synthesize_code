"""Test IR extraction with various function types."""
import sys
sys.path.insert(0, '/workspace/jit/code/extraction')

import jax
import jax.numpy as jnp
from ir_extractor import extract_ir, extract_ir_pair

# Test 1: Simple array addition
def add_function(a, b):
    """Simple addition."""
    return a + b

# Test 2: Reduction (dot product)
def dot_product(a, b):
    """Dot product using reduction."""
    return jnp.sum(a * b)

# Test 3: Scalar + array operations (SAXPY)
def saxpy(alpha, x, y):
    """SAXPY operation."""
    return alpha * x + y

# Test 4: Branching function
def branch_function(a):
    """Function with branching."""
    return jnp.where(a > 0.0, a * 2.0, a * -1.0)

# Test 5: Loop function (using scan)
def loop_function(a):
    """Function with loop using scan."""
    def body_fun(carry, _):
        return carry + a, None
    result, _ = jax.lax.scan(body_fun, jnp.zeros_like(a), jnp.arange(5))
    return result

# Test 6: Vector operations
def vec_function(a, b):
    """Vector dot product."""
    return jnp.sum(a * b, axis=-1)


def run_tests():
    functions = [
        ("add_function", add_function, [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
        ("dot_product", dot_product, [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
        ("saxpy", saxpy, [2.0, jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
        ("branch_function", branch_function, [jnp.array([1.0, -2.0, 3.0])]),
        ("loop_function", loop_function, [jnp.array([1.0, 2.0])]),
        ("vec_function", vec_function, [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]),
    ]
    
    results = []
    for name, func, inputs in functions:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            ir = extract_ir(func, *inputs)
            py_src, jaxpr_code, hlo_code = ir.python_source, ir.jaxpr_code, ir.hlo_code
            
            print(f"Python source length: {len(py_src)} chars")
            print(f"JAXPR code length: {len(jaxpr_code)} chars")
            print(f"HLO code length: {len(hlo_code) if hlo_code else 0} chars")
            print(f"Has HLO: {hlo_code is not None}")
            
            results.append((name, True, len(py_src), len(jaxpr_code)))
            
            # Print first part of Python source
            print(f"\nPython source:")
            print(py_src[:300])
            
            # Print JAXPR
            print(f"\nJAXPR (first 300 chars):")
            print(jaxpr_code[:300])
            
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
    
    for name, success, py_len, jaxpr_len in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}: py={py_len}, jaxpr={jaxpr_len}")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
