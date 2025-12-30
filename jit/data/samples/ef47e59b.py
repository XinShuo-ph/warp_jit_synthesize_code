import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.exp(x)
    v2 = jnp.subtract(v1, y)
    v3 = jnp.cos(v2)
    v4 = jnp.cos(v3)
    v5 = jnp.maximum(v2, v1)
    return v5