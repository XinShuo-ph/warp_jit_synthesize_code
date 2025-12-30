import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, y)
    v1 = jnp.exp(y)
    v2 = jnp.minimum(y, v1)
    v3 = jnp.add(v2, v0)
    v4 = jnp.maximum(y, v0)
    v5 = jnp.subtract(v0, v4)
    v6 = jnp.subtract(v2, v5)
    v7 = jnp.add(v1, x)
    return v7