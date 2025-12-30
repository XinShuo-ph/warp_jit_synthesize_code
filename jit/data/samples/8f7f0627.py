import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.minimum(y, v0)
    v2 = jnp.subtract(y, v0)
    v3 = jnp.square(x)
    v4 = jnp.maximum(v2, y)
    v5 = jnp.cos(v1)
    return v5