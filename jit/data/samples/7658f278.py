import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.add(v0, y)
    v2 = jnp.multiply(v1, x)
    v3 = jnp.abs(v2)
    v4 = jnp.maximum(v0, v1)
    v5 = jnp.multiply(v4, y)
    v6 = jnp.minimum(v4, v1)
    v7 = jnp.exp(y)
    v8 = jnp.minimum(y, v0)
    return v8