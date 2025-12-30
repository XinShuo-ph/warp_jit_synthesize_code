"""Simple JAX jit test."""

import jax
import jax.numpy as jnp


@jax.jit
def add_kernel(a, b):
    return a + b


if __name__ == "__main__":
    n = 10
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.arange(n, dtype=jnp.float32)

    c = add_kernel(a, b).block_until_ready()
    print("Result:", list(c))
    print("Expected:", [float(i * 2) for i in range(n)])
    print("Function compiled and executed successfully!")
