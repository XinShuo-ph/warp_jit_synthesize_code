import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.cos(y)
    v2 = jnp.sin(v0)
    v3 = jnp.exp(v2)
    v4 = jnp.abs(y)
    v5 = jnp.maximum(v3, v1)
    v6 = jnp.exp(v5)
    return v6