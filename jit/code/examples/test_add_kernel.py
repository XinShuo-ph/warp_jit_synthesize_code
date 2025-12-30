"""Simple JAX kernel test (elementwise add)."""

import jax
import jax.numpy as jnp


def add_kernel(a, b):
    return a + b

if __name__ == "__main__":
    n = 10
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.arange(n, dtype=jnp.float32)

    out = jax.jit(add_kernel)(a, b)
    print("Result:", list(out))
    print("Expected:", [float(i*2) for i in range(n)])
    print("Kernel compiled and executed successfully!")
