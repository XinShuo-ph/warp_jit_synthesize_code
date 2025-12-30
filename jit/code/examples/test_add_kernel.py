"""Simple JAX kernel test."""
import jax
import jax.numpy as jnp

# Enable JIT compilation
jax.config.update('jax_enable_x64', True)

@jax.jit
def add_kernel(a, b):
    """Elementwise addition kernel."""
    return a + b

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)
    b = jnp.array([float(i) for i in range(n)], dtype=jnp.float64)
    
    c = add_kernel(a, b)
    
    print("Result:", c)
    print("Expected:", [float(i*2) for i in range(n)])
    print("Match:", jnp.allclose(c, jnp.array([float(i*2) for i in range(n)])))
    print("Kernel compiled and executed successfully!")
