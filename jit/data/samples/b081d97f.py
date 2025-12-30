import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.multiply(x, y)
    v2 = jnp.cos(x)
    return v2