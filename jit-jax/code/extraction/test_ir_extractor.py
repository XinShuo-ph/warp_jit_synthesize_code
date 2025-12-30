"""Test cases for IR extraction - 5 Python→IR pairs."""
import sys
sys.path.insert(0, '/workspace/jit-jax/code/extraction')

import jax
import jax.numpy as jnp
import jax.lax as lax
from ir_extractor import extract_ir, create_ir_pair


# Test Case 1: Simple Arithmetic
def arithmetic_fn(x, y):
    """Element-wise arithmetic: x^2 + 2*y - 1."""
    return x**2 + 2*y - 1


# Test Case 2: Matrix Operations
def matrix_fn(A, x):
    """Matrix-vector multiply with activation: relu(Ax)."""
    return jnp.maximum(0, A @ x)


# Test Case 3: Gradient Computation
def loss_fn(w, x, y):
    """MSE loss function."""
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)


# Test Case 4: Control Flow (jax.lax.cond)
def cond_fn(x):
    """Conditional: if sum(x) > 0 then x*2 else x/2."""
    return lax.cond(
        jnp.sum(x) > 0,
        lambda x: x * 2,
        lambda x: x / 2,
        x
    )


# Test Case 5: Scan (loop) Operations
def scan_fn(xs):
    """Cumulative sum using scan."""
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry
    
    _, cumsum = lax.scan(step, 0.0, xs)
    return cumsum


def run_test_case(name: str, fn, *args, format: str = "stablehlo"):
    """Run a test case and display results."""
    print(f"\n{'='*60}")
    print(f"Test Case: {name}")
    print(f"{'='*60}")
    
    # Create IR pair
    pair = create_ir_pair(fn, *args, format=format)
    
    print(f"Function: {pair['function_name']}")
    print(f"Args: {pair['arg_info']}")
    print(f"\nPython Source:")
    print(pair.get('python', 'N/A'))
    print(f"\n{format.upper()} IR:")
    print(pair['ir'])
    
    return pair


def main():
    key = jax.random.PRNGKey(42)
    
    # Test Case 1: Simple Arithmetic
    x1 = jnp.array([1.0, 2.0, 3.0])
    y1 = jnp.array([0.5, 1.0, 1.5])
    pair1 = run_test_case("Simple Arithmetic", arithmetic_fn, x1, y1)
    
    # Test Case 2: Matrix Operations
    A = jax.random.normal(key, (3, 4))
    x2 = jax.random.normal(key, (4,))
    pair2 = run_test_case("Matrix Operations", matrix_fn, A, x2)
    
    # Test Case 3: Gradient (extract gradient function IR)
    grad_loss = jax.grad(loss_fn)
    w = jax.random.normal(key, (4,))
    X = jax.random.normal(key, (8, 4))
    y = jax.random.normal(key, (8,))
    pair3 = run_test_case("Gradient Computation", grad_loss, w, X, y)
    
    # Test Case 4: Control Flow
    x4 = jnp.array([1.0, -0.5, 0.5])
    pair4 = run_test_case("Control Flow (cond)", cond_fn, x4)
    
    # Test Case 5: Scan Operations
    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pair5 = run_test_case("Scan (loop)", scan_fn, xs)
    
    print("\n" + "="*60)
    print("All 5 test cases completed successfully!")
    print("="*60)
    
    return [pair1, pair2, pair3, pair4, pair5]


if __name__ == "__main__":
    pairs = main()
    
    # Run twice for consistency check
    print("\n\nRunning consistency check (2nd run)...")
    pairs2 = main()
    
    # Verify IR is consistent
    for i, (p1, p2) in enumerate(zip(pairs, pairs2)):
        assert p1['ir'] == p2['ir'], f"Test case {i+1} IR mismatch!"
    
    print("\n✓ Consistency check passed - all IRs match between runs!")
