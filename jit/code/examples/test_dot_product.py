"""Dot product kernel test using JAX."""
import jax
import jax.numpy as jnp

# Enable JIT compilation
jax.config.update('jax_enable_x64', True)

@jax.jit
def dot_product(a, b):
    """Compute dot product of two vectors."""
    return jnp.sum(a * b)

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)  # 0, 1, 2, ..., 9
    b = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)  # 0, 1, 2, ..., 9
    
    result = dot_product(a, b)
    
    computed = float(result)
    expected = sum(i*i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
