"""Simple JAX function test."""
import jax
import jax.numpy as jnp


def add_kernel(a, b):
    """Elementwise addition."""
    return a + b


if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)], dtype=jnp.float32)
    b = jnp.array([float(i) for i in range(n)], dtype=jnp.float32)
    
    # JIT compile for performance
    jitted_add = jax.jit(add_kernel)
    
    c = jitted_add(a, b)
    print("Result:", c)
    print("Expected:", [float(i*2) for i in range(n)])
    print("Kernel compiled and executed successfully!")
