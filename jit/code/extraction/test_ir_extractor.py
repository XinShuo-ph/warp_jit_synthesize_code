"""Test cases for IR extraction with diverse function types."""
import jax
import jax.numpy as jnp
from ir_extractor import extract_all, save_pair
import os


# Test Case 1: Arithmetic operations
def arithmetic_ops(a, b, c):
    """Basic arithmetic: add, sub, mul, div."""
    return (a + b) * c - (a / b)


# Test Case 2: Trigonometric functions
def trig_functions(x):
    """Trigonometric operations."""
    return jnp.sin(x) * jnp.cos(x) + jnp.tan(x / 2)


# Test Case 3: Reduction operations
def reductions(x):
    """Reduction operations: sum, mean, max, min."""
    return jnp.sum(x) + jnp.mean(x) + jnp.max(x) - jnp.min(x)


# Test Case 4: Matrix operations
def matrix_ops(A, B):
    """Matrix operations: matmul, transpose."""
    return jnp.dot(A, B.T) + jnp.eye(A.shape[0])


# Test Case 5: Control flow with where
def conditional(x, threshold):
    """Conditional selection using where."""
    return jnp.where(x > threshold, x, threshold)


# Test Case 6: Exponential/logarithmic
def exp_log_ops(x):
    """Exponential and logarithmic operations."""
    return jnp.exp(x) + jnp.log(jnp.abs(x) + 1) + jnp.sqrt(jnp.abs(x))


# Test Case 7: Neural network layer (dense)
def dense_layer(W, x, b):
    """Dense/fully-connected layer."""
    return jnp.tanh(jnp.dot(W, x) + b)


# Test Case 8: Batch normalization-like
def normalize(x):
    """Normalize to zero mean, unit variance."""
    mean = jnp.mean(x)
    std = jnp.std(x) + 1e-6
    return (x - mean) / std


def run_tests():
    """Run all test cases and verify extraction works."""
    # Prepare test inputs
    x4 = jnp.array([1.0, 2.0, 3.0, 4.0])
    y4 = jnp.array([0.5, 1.0, 1.5, 2.0])
    z4 = jnp.array([2.0, 2.0, 2.0, 2.0])
    mat2 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    vec2 = jnp.array([1.0, 2.0])
    scalar = 1.5
    
    test_cases = [
        ("arithmetic_ops", arithmetic_ops, (x4, y4, z4)),
        ("trig_functions", trig_functions, (x4,)),
        ("reductions", reductions, (x4,)),
        ("matrix_ops", matrix_ops, (mat2, mat2)),
        ("conditional", conditional, (x4, scalar)),
        ("exp_log_ops", exp_log_ops, (x4,)),
        ("dense_layer", dense_layer, (mat2, vec2, vec2)),
        ("normalize", normalize, (x4,)),
    ]
    
    results = []
    print("Running IR extraction tests...\n")
    
    for name, fn, args in test_cases:
        try:
            data = extract_all(fn, *args)
            results.append(data)
            print(f"✓ {name}")
            print(f"  JAXPR: {data['jaxpr'][:80]}...")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    print(f"\nPassed: {len(results)}/{len(test_cases)}")
    return results


def save_test_pairs(output_dir: str):
    """Save all test cases as JSON pairs."""
    os.makedirs(output_dir, exist_ok=True)
    results = run_tests()
    
    for data in results:
        filepath = os.path.join(output_dir, f"{data['name']}.json")
        save_pair(data, filepath)
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    run_tests()
