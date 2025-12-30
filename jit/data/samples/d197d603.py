import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.multiply(y, y)
    v2 = jnp.exp(x)
    return v2