import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.abs(x)
    v2 = jnp.maximum(v1, y)
    return v2