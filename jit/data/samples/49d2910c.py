import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.minimum(y, v0)
    v2 = jnp.multiply(x, v1)
    v3 = jnp.maximum(v2, x)
    v4 = jnp.maximum(x, v2)
    v5 = jnp.add(x, x)
    v6 = jnp.exp(v1)
    v7 = jnp.abs(y)
    return v7