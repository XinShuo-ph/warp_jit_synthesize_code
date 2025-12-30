import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.multiply(v1, v1)
    v3 = jnp.maximum(x, x)
    v4 = jnp.maximum(y, x)
    v5 = jnp.multiply(v4, y)
    v6 = jnp.maximum(y, v1)
    return v6