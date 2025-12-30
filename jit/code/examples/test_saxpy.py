"""SAXPY (Single-precision A*X Plus Y) test using JAX."""

import jax
import jax.numpy as jnp


@jax.jit
def saxpy(a: float, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return a * x + y

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = jnp.arange(n, dtype=jnp.float32)
    y = (jnp.arange(n, dtype=jnp.float32) * 10.0).astype(jnp.float32)

    out = saxpy(a, x, y)
    result = jnp.asarray(out).block_until_ready()
    expected = [a * i + i * 10 for i in range(n)]
    print(f"SAXPY result: {result.tolist()}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(r - e) < 1e-6 for r, e in zip(result.tolist(), expected))}")
