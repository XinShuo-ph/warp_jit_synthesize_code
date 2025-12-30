import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.maximum(x, v0)
    v2 = jnp.cos(y)
    v3 = jnp.cos(v0)
    v4 = jnp.sin(v2)
    v5 = jnp.add(v0, v2)
    v6 = jnp.abs(v0)
    v7 = jnp.maximum(v4, x)
    return v7