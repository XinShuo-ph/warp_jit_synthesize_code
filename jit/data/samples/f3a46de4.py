import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.maximum(x, y)
    v2 = jnp.add(x, v0)
    v3 = jnp.multiply(v2, v2)
    v4 = jnp.sin(v3)
    v5 = jnp.minimum(v0, x)
    v6 = jnp.maximum(x, v4)
    v7 = jnp.sin(v2)
    return v7