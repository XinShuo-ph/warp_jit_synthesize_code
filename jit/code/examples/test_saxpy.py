"""SAXPY (Single-precision A*X Plus Y) JAX JIT example."""

import jax
import jax.numpy as jnp


@jax.jit
def saxpy(a: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return a * x + y


if __name__ == "__main__":
    n = 8
    a = jnp.asarray(2.0, dtype=jnp.float32)
    x = jnp.arange(n, dtype=jnp.float32)
    y = jnp.arange(n, dtype=jnp.float32) * 10.0

    out = saxpy(a, x, y)
    result = jax.device_get(out)

    expected = [float(2.0 * i + i * 10) for i in range(n)]
    print(f"SAXPY result: {list(result)}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(float(r) - e) < 1e-6 for r, e in zip(result, expected))}")
