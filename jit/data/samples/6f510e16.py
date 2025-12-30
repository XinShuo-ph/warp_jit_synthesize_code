import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.abs(v0)
    v2 = jnp.cos(x)
    v3 = jnp.square(v0)
    v4 = jnp.maximum(v0, y)
    return v4