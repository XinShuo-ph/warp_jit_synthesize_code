import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.multiply(x, y)
    v2 = jnp.sin(y)
    v3 = jnp.subtract(y, x)
    return v3