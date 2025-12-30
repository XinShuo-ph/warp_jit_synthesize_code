"""Dot product JAX JIT example (reduction)."""

import jax
import jax.numpy as jnp


@jax.jit
def dot_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(a * b)


if __name__ == "__main__":
    n = 10
    a = jnp.arange(n, dtype=jnp.float32)  # 0, 1, 2, ..., 9
    b = jnp.arange(n, dtype=jnp.float32)  # 0, 1, 2, ..., 9

    computed = float(jax.device_get(dot_product(a, b)))
    expected = sum(i * i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
