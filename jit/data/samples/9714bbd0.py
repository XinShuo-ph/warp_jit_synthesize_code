import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.add(x, y)
    v2 = jnp.sin(v0)
    v3 = jnp.subtract(x, x)
    v4 = jnp.minimum(v0, v0)
    v5 = jnp.sin(y)
    v6 = jnp.subtract(v1, y)
    v7 = jnp.minimum(v3, y)
    return v7