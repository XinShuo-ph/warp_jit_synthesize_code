import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.sin(x)
    v2 = jnp.sin(v0)
    v3 = jnp.square(v2)
    v4 = jnp.maximum(v3, y)
    v5 = jnp.subtract(y, v2)
    v6 = jnp.subtract(v0, y)
    v7 = jnp.subtract(v4, v5)
    v8 = jnp.cos(v2)
    v9 = jnp.abs(y)
    return v9