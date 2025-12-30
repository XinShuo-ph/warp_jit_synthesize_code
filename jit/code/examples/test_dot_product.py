"""Dot product kernel test (reduction) using JAX."""

import jax
import jax.numpy as jnp


def dot_product(a, b):
    return jnp.sum(a * b)

if __name__ == "__main__":
    n = 10
    a = jnp.arange(n, dtype=jnp.float32)  # 0, 1, 2, ..., 9
    b = jnp.arange(n, dtype=jnp.float32)  # 0, 1, 2, ..., 9

    computed = float(jax.jit(dot_product)(a, b))
    expected = sum(i*i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
