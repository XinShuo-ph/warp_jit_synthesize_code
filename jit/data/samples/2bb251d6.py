import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.cos(v0)
    v2 = jnp.maximum(v1, x)
    v3 = jnp.cos(v1)
    v4 = jnp.sin(v0)
    v5 = jnp.multiply(v1, v3)
    v6 = jnp.add(y, y)
    v7 = jnp.add(y, v4)
    return v7