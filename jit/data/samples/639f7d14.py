import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.maximum(v0, y)
    v2 = jnp.maximum(x, v0)
    return v2