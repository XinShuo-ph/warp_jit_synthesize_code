import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.abs(x)
    v2 = jnp.abs(y)
    v3 = jnp.exp(v1)
    v4 = jnp.abs(x)
    v5 = jnp.minimum(v4, v4)
    v6 = jnp.multiply(v5, y)
    return v6