import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(y)
    v1 = jnp.maximum(v0, v0)
    v2 = jnp.cos(x)
    v3 = jnp.multiply(y, y)
    v4 = jnp.maximum(y, y)
    v5 = jnp.cos(v1)
    v6 = jnp.add(v3, v1)
    return v6