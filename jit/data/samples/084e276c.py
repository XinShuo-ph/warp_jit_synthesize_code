import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.maximum(v0, v0)
    v2 = jnp.minimum(v0, v0)
    return v2