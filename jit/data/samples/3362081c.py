import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.square(y)
    v2 = jnp.maximum(y, y)
    return v2