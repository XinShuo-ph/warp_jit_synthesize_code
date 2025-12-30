import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.maximum(x, v0)
    v2 = jnp.minimum(x, v1)
    v3 = jnp.add(v0, x)
    v4 = jnp.minimum(y, v1)
    return v4