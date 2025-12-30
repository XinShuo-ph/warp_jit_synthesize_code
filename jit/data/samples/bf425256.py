import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.multiply(x, y)
    v2 = jnp.minimum(v1, x)
    v3 = jnp.maximum(v2, v0)
    v4 = jnp.cos(v2)
    v5 = jnp.square(v0)
    v6 = jnp.exp(v1)
    v7 = jnp.minimum(v6, x)
    return v7