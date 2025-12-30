import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.square(y)
    v2 = jnp.abs(v1)
    v3 = jnp.exp(v1)
    v4 = jnp.cos(v0)
    v5 = jnp.cos(v4)
    v6 = jnp.maximum(v0, v0)
    v7 = jnp.multiply(v1, v1)
    v8 = jnp.subtract(v7, v7)
    v9 = jnp.exp(v1)
    return v9