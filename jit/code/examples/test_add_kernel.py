"""Simple JAX JIT example: elementwise add."""

import jax
import jax.numpy as jnp


@jax.jit
def add_kernel(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b


if __name__ == "__main__":
    n = 10
    a = jnp.arange(n, dtype=jnp.float32)
    b = jnp.arange(n, dtype=jnp.float32)

    c = add_kernel(a, b)
    result = jax.device_get(c)

    print("Result:", result)
    print("Expected:", [float(i * 2) for i in range(n)])
    print("Kernel compiled and executed successfully!")
