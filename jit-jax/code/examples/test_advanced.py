"""Validation tests for advanced JAX operations."""
import jax
import jax.numpy as jnp
import jax.lax as lax


def test_vmap_vs_manual_loop():
    """Test that vmap produces same results as manual loop."""
    key = jax.random.PRNGKey(123)
    
    def single_fn(x):
        return jnp.sin(x) + x**2
    
    batch = jax.random.normal(key, (10, 4))
    
    # vmap version
    vmapped = jax.vmap(single_fn)(batch)
    
    # Manual loop version
    manual = jnp.stack([single_fn(batch[i]) for i in range(batch.shape[0])])
    
    assert jnp.allclose(vmapped, manual, rtol=1e-5), "vmap mismatch with manual loop"
    print("✓ test_vmap_vs_manual_loop passed")


def test_grad_vs_numerical():
    """Test that grad matches numerical differentiation."""
    def f(x):
        return jnp.sin(x) * jnp.exp(-x**2)
    
    grad_f = jax.grad(f)
    
    x = 1.0
    eps = 1e-5
    
    # Analytical gradient
    analytical = grad_f(x)
    
    # Numerical gradient (central difference)
    numerical = (f(x + eps) - f(x - eps)) / (2 * eps)
    
    assert jnp.isclose(analytical, numerical, rtol=1e-2), \
        f"Gradient mismatch: analytical={analytical}, numerical={numerical}"
    print(f"✓ test_grad_vs_numerical passed (analytical={analytical:.6f}, numerical={numerical:.6f})")


def test_grad_vector_function():
    """Test gradient of vector-to-scalar function."""
    def f(x):
        return jnp.sum(x**3)
    
    grad_f = jax.grad(f)
    
    x = jnp.array([1.0, 2.0, 3.0])
    
    # Analytical: d/dx (sum(x^3)) = 3x^2
    expected = 3 * x**2
    actual = grad_f(x)
    
    assert jnp.allclose(actual, expected), f"Gradient mismatch: {actual} vs {expected}"
    print("✓ test_grad_vector_function passed")


def test_second_derivative():
    """Test second derivative computation."""
    def f(x):
        return x**4
    
    # f'(x) = 4x^3, f''(x) = 12x^2
    grad2_f = jax.grad(jax.grad(f))
    
    x = 2.0
    expected = 12 * x**2  # 48
    actual = grad2_f(x)
    
    assert jnp.isclose(actual, expected), f"Second derivative mismatch: {actual} vs {expected}"
    print(f"✓ test_second_derivative passed (f''(2) = {actual})")


def test_scan_cumsum():
    """Test scan-based cumulative sum against jnp.cumsum."""
    def scan_cumsum(xs):
        def step(carry, x):
            new_carry = carry + x
            return new_carry, new_carry
        _, result = lax.scan(step, 0.0, xs)
        return result
    
    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    scan_result = scan_cumsum(xs)
    numpy_result = jnp.cumsum(xs)
    
    assert jnp.allclose(scan_result, numpy_result), "Scan cumsum mismatch"
    print("✓ test_scan_cumsum passed")


def test_jit_consistency():
    """Test that jitted function produces consistent results."""
    @jax.jit
    def complex_fn(x, y):
        return jnp.dot(jnp.sin(x), jnp.cos(y)) + jnp.sum(x * y)
    
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (10,))
    y = jax.random.normal(key, (10,))
    
    # Run multiple times
    results = [complex_fn(x, y) for _ in range(5)]
    
    for i, r in enumerate(results[1:], 1):
        assert jnp.isclose(results[0], r), f"JIT result mismatch at iteration {i}"
    
    print("✓ test_jit_consistency passed")


def test_value_and_grad():
    """Test value_and_grad returns both correctly."""
    def f(x):
        return jnp.sum(x**2)
    
    vg = jax.value_and_grad(f)
    
    x = jnp.array([1.0, 2.0, 3.0])
    value, grad = vg(x)
    
    expected_value = 1.0 + 4.0 + 9.0  # 14
    expected_grad = 2 * x  # [2, 4, 6]
    
    assert jnp.isclose(value, expected_value), f"Value mismatch: {value} vs {expected_value}"
    assert jnp.allclose(grad, expected_grad), f"Grad mismatch: {grad} vs {expected_grad}"
    print("✓ test_value_and_grad passed")


def run_all_tests():
    """Run all tests twice for consistency."""
    print("=" * 60)
    print("Running validation tests...")
    print("=" * 60 + "\n")
    
    tests = [
        test_vmap_vs_manual_loop,
        test_grad_vs_numerical,
        test_grad_vector_function,
        test_second_derivative,
        test_scan_cumsum,
        test_jit_consistency,
        test_value_and_grad,
    ]
    
    for test in tests:
        test()
    
    print(f"\n{'='*60}")
    print(f"All {len(tests)} tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run twice for consistency
    run_all_tests()
    print("\n--- Run 2 ---")
    run_all_tests()
    print("\n✓ Both runs passed - results are consistent!")
