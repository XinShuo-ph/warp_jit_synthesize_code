import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.minimum(x, y)
    v2 = jnp.subtract(y, v1)
    v3 = jnp.sin(y)
    v4 = jnp.add(v0, v2)
    v5 = jnp.exp(v3)
    return v5