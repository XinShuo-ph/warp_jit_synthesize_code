import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.multiply(y, v0)
    v2 = jnp.maximum(y, v1)
    v3 = jnp.cos(v0)
    v4 = jnp.add(v0, v3)
    v5 = jnp.minimum(v0, v0)
    v6 = jnp.cos(v2)
    v7 = jnp.maximum(x, y)
    return v7