import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.maximum(x, v0)
    v2 = jnp.maximum(y, y)
    v3 = jnp.add(v2, v2)
    v4 = jnp.multiply(x, v3)
    v5 = jnp.cos(x)
    return v5