import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.maximum(x, x)
    v2 = jnp.add(x, v0)
    v3 = jnp.cos(v1)
    v4 = jnp.maximum(y, y)
    v5 = jnp.multiply(x, v4)
    return v5