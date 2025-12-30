import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.abs(x)
    v2 = jnp.cos(v0)
    v3 = jnp.cos(v2)
    v4 = jnp.abs(v3)
    v5 = jnp.maximum(v2, v0)
    v6 = jnp.subtract(v4, x)
    v7 = jnp.multiply(v3, x)
    return v7