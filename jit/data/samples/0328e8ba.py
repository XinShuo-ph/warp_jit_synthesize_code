import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.cos(v0)
    v2 = jnp.square(y)
    v3 = jnp.maximum(v2, x)
    return v3