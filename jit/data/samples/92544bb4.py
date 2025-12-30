import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.maximum(v0, y)
    v2 = jnp.square(v1)
    return v2