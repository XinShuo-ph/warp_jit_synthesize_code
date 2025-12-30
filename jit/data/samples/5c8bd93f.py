import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.abs(v0)
    v2 = jnp.multiply(x, x)
    v3 = jnp.maximum(v0, v1)
    v4 = jnp.multiply(y, v1)
    v5 = jnp.subtract(v4, y)
    v6 = jnp.add(v4, v4)
    v7 = jnp.subtract(v6, v4)
    return v7