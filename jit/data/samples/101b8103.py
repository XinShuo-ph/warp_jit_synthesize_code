import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.subtract(v0, y)
    v2 = jnp.maximum(v0, y)
    return v2