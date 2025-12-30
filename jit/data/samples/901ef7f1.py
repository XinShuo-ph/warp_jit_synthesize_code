import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.sin(y)
    v2 = jnp.exp(v1)
    v3 = jnp.multiply(x, y)
    v4 = jnp.subtract(v2, v2)
    v5 = jnp.minimum(v3, v2)
    v6 = jnp.maximum(v2, v0)
    v7 = jnp.cos(v5)
    return v7