import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.maximum(y, y)
    v2 = jnp.subtract(v1, y)
    v3 = jnp.abs(v0)
    v4 = jnp.minimum(x, x)
    v5 = jnp.minimum(v4, v3)
    return v5