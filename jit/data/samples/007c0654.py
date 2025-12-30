import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.cos(v0)
    v2 = jnp.minimum(y, v0)
    v3 = jnp.exp(x)
    v4 = jnp.abs(v2)
    v5 = jnp.exp(v1)
    v6 = jnp.abs(v2)
    v7 = jnp.sin(v4)
    v8 = jnp.sin(y)
    return v8