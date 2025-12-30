"""SAXPY (Single-precision A*X Plus Y) kernel test using JAX."""
import jax
import jax.numpy as jnp

# Enable JIT compilation
jax.config.update('jax_enable_x64', True)

@jax.jit
def saxpy(a, x, y):
    """Compute a * x + y (SAXPY operation)."""
    return a * x + y

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)
    y = jnp.array([float(i * 10) for i in range(n)], dtype=jnp.float64)
    
    out = saxpy(a, x, y)
    
    result = out
    expected = jnp.array([a * i + i * 10 for i in range(n)])
    print(f"SAXPY result: {list(result)}")
    print(f"Expected: {list(expected)}")
    print(f"Match: {jnp.allclose(result, expected)}")
