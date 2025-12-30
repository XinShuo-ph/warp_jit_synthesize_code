import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.maximum(v0, x)
    v2 = jnp.maximum(y, v1)
    v3 = jnp.minimum(v0, v0)
    return v3