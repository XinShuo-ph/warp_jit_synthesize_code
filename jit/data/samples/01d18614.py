import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.minimum(x, x)
    v2 = jnp.minimum(x, v0)
    v3 = jnp.cos(v1)
    v4 = jnp.square(v1)
    v5 = jnp.multiply(v1, v4)
    v6 = jnp.sin(x)
    v7 = jnp.square(v1)
    return v7