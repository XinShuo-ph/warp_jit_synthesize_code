import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.minimum(y, y)
    v2 = jnp.sin(y)
    v3 = jnp.cos(v2)
    v4 = jnp.sin(x)
    v5 = jnp.cos(v4)
    v6 = jnp.minimum(y, v2)
    return v6