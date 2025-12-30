import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.subtract(x, x)
    v2 = jnp.minimum(y, y)
    v3 = jnp.maximum(v1, y)
    return v3