import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.multiply(v0, x)
    v2 = jnp.abs(v0)
    v3 = jnp.minimum(y, y)
    v4 = jnp.maximum(y, v3)
    v5 = jnp.maximum(v1, v1)
    v6 = jnp.multiply(x, x)
    v7 = jnp.abs(v4)
    return v7