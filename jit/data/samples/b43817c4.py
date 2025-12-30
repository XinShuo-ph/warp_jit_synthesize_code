import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.subtract(y, y)
    v2 = jnp.sin(x)
    v3 = jnp.cos(v1)
    v4 = jnp.abs(v1)
    v5 = jnp.sin(y)
    v6 = jnp.maximum(v5, v1)
    v7 = jnp.cos(x)
    return v7