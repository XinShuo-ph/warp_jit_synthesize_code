import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.minimum(v0, x)
    v2 = jnp.square(v1)
    return v2