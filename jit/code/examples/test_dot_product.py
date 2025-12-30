"""Test dot product with JAX."""
import jax
import jax.numpy as jnp


def dot_product_kernel(a, b):
    """Compute dot product along last axis."""
    return jnp.sum(a * b, axis=-1)


if __name__ == "__main__":
    n = 10
    vec_dim = 3
    
    # Create 2D arrays where last dimension is vector dimension
    a = jnp.ones((n, vec_dim), dtype=jnp.float32)
    b = jnp.ones((n, vec_dim), dtype=jnp.float32) * 2.0
    
    # JIT compile
    jitted_dot = jax.jit(dot_product_kernel)
    
    result = jitted_dot(a, b)
    print("Result shape:", result.shape)
    print("Result:", result)
    print("Expected: [6.0, 6.0, ...]")
    print("Kernel compiled and executed successfully!")
