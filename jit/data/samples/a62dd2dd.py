import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.maximum(y, x)
    v2 = jnp.minimum(x, y)
    v3 = jnp.maximum(x, y)
    return v3