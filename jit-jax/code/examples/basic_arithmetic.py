"""Basic arithmetic operations with JAX JIT compilation."""
import jax
import jax.numpy as jnp


@jax.jit
def vector_add(x, y):
    """Add two vectors element-wise."""
    return x + y


@jax.jit
def polynomial(x, a, b, c):
    """Compute ax^2 + bx + c."""
    return a * x**2 + b * x + c


@jax.jit
def compound_ops(x):
    """Multiple arithmetic operations."""
    return jnp.sin(x) * jnp.cos(x) + jnp.exp(-x**2)


def main():
    # Test vector_add
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    result1 = vector_add(x, y)
    print(f"vector_add: {result1}")
    
    # Test polynomial
    x = jnp.array([0.0, 1.0, 2.0])
    result2 = polynomial(x, 2.0, 3.0, 1.0)  # 2x^2 + 3x + 1
    print(f"polynomial: {result2}")
    
    # Test compound_ops
    x = jnp.array([0.0, 0.5, 1.0])
    result3 = compound_ops(x)
    print(f"compound_ops: {result3}")
    
    return result1, result2, result3


if __name__ == "__main__":
    r1, r2, r3 = main()
    print("\nRun 2:")
    r1_2, r2_2, r3_2 = main()
    
    # Verify consistency
    assert jnp.allclose(r1, r1_2)
    assert jnp.allclose(r2, r2_2)
    assert jnp.allclose(r3, r3_2)
    print("\nConsistency check passed!")
