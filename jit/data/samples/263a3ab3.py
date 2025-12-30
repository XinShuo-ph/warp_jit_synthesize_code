import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.abs(v0)
    v2 = jnp.multiply(v1, v0)
    v3 = jnp.add(y, x)
    v4 = jnp.subtract(v1, v3)
    v5 = jnp.cos(x)
    v6 = jnp.maximum(v2, v0)
    v7 = jnp.multiply(v6, y)
    return v7