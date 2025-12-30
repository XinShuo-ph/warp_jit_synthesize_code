import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.add(y, y)
    v2 = jnp.subtract(v0, v1)
    v3 = jnp.sin(v1)
    v4 = jnp.cos(y)
    v5 = jnp.minimum(v1, v1)
    v6 = jnp.maximum(v2, x)
    return v6