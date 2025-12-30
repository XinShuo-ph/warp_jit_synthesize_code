import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.cos(y)
    v2 = jnp.square(v0)
    v3 = jnp.multiply(v2, x)
    v4 = jnp.maximum(v0, v2)
    return v4