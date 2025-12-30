import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.maximum(v1, y)
    v3 = jnp.sin(y)
    v4 = jnp.minimum(x, v2)
    v5 = jnp.exp(x)
    v6 = jnp.cos(v3)
    v7 = jnp.subtract(v4, v0)
    v8 = jnp.maximum(v1, y)
    return v8