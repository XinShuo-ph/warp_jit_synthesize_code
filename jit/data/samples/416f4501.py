import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.abs(v0)
    v2 = jnp.maximum(v1, v1)
    v3 = jnp.multiply(y, y)
    return v3