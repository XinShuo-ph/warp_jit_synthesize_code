import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.add(y, x)
    v2 = jnp.cos(v1)
    v3 = jnp.square(v1)
    v4 = jnp.square(v3)
    v5 = jnp.add(x, v2)
    v6 = jnp.maximum(v4, y)
    v7 = jnp.multiply(y, v4)
    return v7