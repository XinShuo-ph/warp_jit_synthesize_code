import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.square(v0)
    v2 = jnp.multiply(y, x)
    v3 = jnp.abs(y)
    v4 = jnp.exp(y)
    v5 = jnp.maximum(v1, v2)
    v6 = jnp.subtract(x, y)
    v7 = jnp.subtract(y, v4)
    return v7