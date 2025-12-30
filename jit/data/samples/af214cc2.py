import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.maximum(y, y)
    v2 = jnp.sin(v1)
    v3 = jnp.cos(v2)
    v4 = jnp.sin(v3)
    v5 = jnp.add(v4, v4)
    return v5