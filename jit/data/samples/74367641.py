import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.multiply(v0, y)
    v3 = jnp.square(v0)
    v4 = jnp.cos(v1)
    v5 = jnp.cos(v3)
    v6 = jnp.sin(v0)
    v7 = jnp.cos(v5)
    return v7