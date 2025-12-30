import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.maximum(x, y)
    v2 = jnp.sin(x)
    v3 = jnp.subtract(y, v1)
    return v3