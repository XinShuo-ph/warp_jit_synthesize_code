import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.cos(y)
    v2 = jnp.exp(v1)
    v3 = jnp.sin(y)
    v4 = jnp.maximum(v1, v2)
    v5 = jnp.subtract(v4, x)
    v6 = jnp.add(v1, v1)
    v7 = jnp.add(v0, x)
    v8 = jnp.maximum(v4, v0)
    return v8