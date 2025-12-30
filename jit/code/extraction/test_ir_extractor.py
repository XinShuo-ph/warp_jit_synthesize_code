"""
Test cases for JAX IR extraction
Covers various operation types and patterns
"""

import jax
import jax.numpy as jnp
from jax import lax
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extraction.ir_extractor import IRExtractor


# ============================================================================
# Arithmetic Operations
# ============================================================================

def test_add(x, y):
    """Addition: x + y"""
    return x + y


def test_mul(x, y):
    """Multiplication: x * y"""
    return x * y


def test_sub(x, y):
    """Subtraction: x - y"""
    return x - y


def test_div(x, y):
    """Division: x / y"""
    return x / y


def test_combined_arithmetic(x, y, z):
    """Combined: (x + y) * z - x / y"""
    return (x + y) * z - x / y


# ============================================================================
# Math Functions
# ============================================================================

def test_sin(x):
    """Sine function"""
    return jnp.sin(x)


def test_cos(x):
    """Cosine function"""
    return jnp.cos(x)


def test_exp(x):
    """Exponential function"""
    return jnp.exp(x)


def test_log(x):
    """Natural logarithm"""
    return jnp.log(x)


def test_tanh(x):
    """Hyperbolic tangent"""
    return jnp.tanh(x)


def test_sqrt(x):
    """Square root"""
    return jnp.sqrt(x)


def test_power(x):
    """Power: x^3"""
    return x ** 3


def test_combined_math(x):
    """Combined: tanh(sin(x) + exp(x))"""
    return jnp.tanh(jnp.sin(x) + jnp.exp(x))


# ============================================================================
# Array Operations
# ============================================================================

def test_dot(x, y):
    """Dot product"""
    return jnp.dot(x, y)


def test_matmul(A, B):
    """Matrix multiplication"""
    return jnp.matmul(A, B)


def test_sum(x):
    """Sum all elements"""
    return jnp.sum(x)


def test_mean(x):
    """Mean of all elements"""
    return jnp.mean(x)


def test_transpose(A):
    """Matrix transpose"""
    return A.T


def test_reshape(x):
    """Reshape array"""
    return x.reshape(-1, 1)


def test_reduction(x):
    """Sum along axis"""
    return jnp.sum(x, axis=0)


# ============================================================================
# Control Flow
# ============================================================================

def test_where(x):
    """Conditional: where(x > 0, x^2, -x)"""
    return jnp.where(x > 0, x ** 2, -x)


def test_cond_simple(x):
    """Simple conditional using lax.cond"""
    return lax.cond(
        x > 0,
        lambda x: x ** 2,
        lambda x: -x,
        x
    )


def test_select(pred, x, y):
    """Select based on predicate"""
    return jnp.where(pred, x, y)


# ============================================================================
# Vectorization (vmap)
# ============================================================================

def simple_fn(x):
    """Simple function for vmap"""
    return x ** 2 + 2 * x + 1


def test_vmap():
    """Vectorized function using vmap"""
    vmapped = jax.vmap(simple_fn)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    return vmapped(x)


# ============================================================================
# Complex Patterns
# ============================================================================

def test_linear_layer(W, x, b):
    """Linear layer: Wx + b"""
    return jnp.dot(W, x) + b


def test_relu(x):
    """ReLU activation"""
    return jnp.maximum(0, x)


def test_softmax(x):
    """Softmax function"""
    return jnp.exp(x) / jnp.sum(jnp.exp(x))


def test_layer_norm(x):
    """Layer normalization"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return (x - mean) / jnp.sqrt(var + 1e-5)


def test_mse_loss(pred, target):
    """Mean squared error loss"""
    return jnp.mean((pred - target) ** 2)


# ============================================================================
# Gradient Computation
# ============================================================================

def loss_fn(x):
    """Simple loss for gradient testing"""
    return jnp.sum(x ** 2)


def test_grad():
    """Gradient computation"""
    grad_fn = jax.grad(loss_fn)
    x = jnp.array([1.0, 2.0, 3.0])
    return grad_fn(x)


# ============================================================================
# Main Test Runner
# ============================================================================

def run_tests():
    """Run all test cases and extract IR."""
    print("=" * 80)
    print("JAX IR Extraction - Comprehensive Test Suite")
    print("=" * 80)
    
    extractor = IRExtractor(dialect='stablehlo')
    results = []
    
    # Arithmetic tests
    print("\n1. Arithmetic Operations")
    print("-" * 80)
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = jnp.array([7.0, 8.0, 9.0])
    
    arith_tests = [
        (test_add, (x, y)),
        (test_mul, (x, y)),
        (test_sub, (x, y)),
        (test_div, (x, y)),
        (test_combined_arithmetic, (x, y, z))
    ]
    
    for func, args in arith_tests:
        result = extractor.extract_with_metadata(func, *args)
        results.append(result)
        print(f"  ✓ {result['function_name']}")
    
    # Math function tests
    print("\n2. Math Functions")
    print("-" * 80)
    x_pos = jnp.array([1.0, 2.0, 3.0])
    
    math_tests = [
        (test_sin, (x_pos,)),
        (test_cos, (x_pos,)),
        (test_exp, (x_pos,)),
        (test_log, (x_pos,)),
        (test_tanh, (x_pos,)),
        (test_sqrt, (x_pos,)),
        (test_power, (x_pos,)),
        (test_combined_math, (x_pos,))
    ]
    
    for func, args in math_tests:
        result = extractor.extract_with_metadata(func, *args)
        results.append(result)
        print(f"  ✓ {result['function_name']}")
    
    # Array operation tests
    print("\n3. Array Operations")
    print("-" * 80)
    vec = jnp.array([1.0, 2.0, 3.0])
    mat = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    
    array_tests = [
        (test_dot, (vec, vec)),
        (test_matmul, (mat, mat)),
        (test_sum, (vec,)),
        (test_mean, (vec,)),
        (test_transpose, (mat,)),
        (test_reshape, (vec,)),
        (test_reduction, (mat,))
    ]
    
    for func, args in array_tests:
        result = extractor.extract_with_metadata(func, *args)
        results.append(result)
        print(f"  ✓ {result['function_name']}")
    
    # Control flow tests
    print("\n4. Control Flow")
    print("-" * 80)
    x_mixed = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    pred = jnp.array([True, False, True])
    
    control_tests = [
        (test_where, (x_mixed,)),
        (test_cond_simple, (1.0,)),
        (test_select, (pred, x, y))
    ]
    
    for func, args in control_tests:
        result = extractor.extract_with_metadata(func, *args)
        results.append(result)
        print(f"  ✓ {result['function_name']}")
    
    # Complex pattern tests
    print("\n5. Complex Patterns")
    print("-" * 80)
    W = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x_vec = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([0.5, 1.5])
    
    complex_tests = [
        (test_linear_layer, (W, x_vec, b)),
        (test_relu, (x_mixed,)),
        (test_softmax, (x_pos,)),
        (test_layer_norm, (x_pos,)),
        (test_mse_loss, (x, y))
    ]
    
    for func, args in complex_tests:
        result = extractor.extract_with_metadata(func, *args)
        results.append(result)
        print(f"  ✓ {result['function_name']}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Total tests: {len(results)}")
    print(f"All tests passed!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_tests()
    
    # Print a sample IR
    print("\nSample IR (test_combined_math):")
    print("-" * 80)
    for r in results:
        if r['function_name'] == 'test_combined_math':
            print(r['ir_code'])
            break
