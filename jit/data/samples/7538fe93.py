import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.maximum(v0, x)
    v2 = jnp.maximum(v1, v1)
    v3 = jnp.cos(y)
    v4 = jnp.cos(y)
    return v4