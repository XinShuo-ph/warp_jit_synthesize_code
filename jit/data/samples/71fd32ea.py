import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.maximum(y, y)
    v2 = jnp.abs(v1)
    return v2