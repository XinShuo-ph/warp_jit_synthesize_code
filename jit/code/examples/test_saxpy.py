"""SAXPY (Single-precision A*X Plus Y) kernel test using JAX."""

import jax
import jax.numpy as jnp


def saxpy(a, x, y):
    return a * x + y

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = jnp.arange(n, dtype=jnp.float32)
    y = (jnp.arange(n, dtype=jnp.float32) * 10.0)

    result = jax.jit(saxpy)(jnp.array(a, dtype=jnp.float32), x, y)
    expected = [a * i + i * 10 for i in range(n)]
    print(f"SAXPY result: {list(result)}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(r - e) < 1e-6 for r, e in zip(result, expected))}")
