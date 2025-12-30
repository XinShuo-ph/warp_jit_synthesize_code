"""SAXPY (A*X + Y) test using JAX."""

import jax
import jax.numpy as jnp


@jax.jit
def saxpy(alpha, x, y):
    return alpha * x + y


if __name__ == "__main__":
    n = 8
    alpha = 2.0
    x = jnp.arange(n, dtype=jnp.float32)
    y = (jnp.arange(n, dtype=jnp.float32) * 10.0)

    out = saxpy(alpha, x, y).block_until_ready()

    result = list(out)
    expected = [alpha * i + i * 10 for i in range(n)]
    print(f"SAXPY result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(float(r) - e) < 1e-6 for r, e in zip(result, expected))}")
