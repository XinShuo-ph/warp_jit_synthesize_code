import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.maximum(y, v0)
    v2 = jnp.cos(y)
    v3 = jnp.add(y, v2)
    v4 = jnp.sin(x)
    v5 = jnp.subtract(x, v4)
    return v5