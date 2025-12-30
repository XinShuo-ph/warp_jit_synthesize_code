import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.add(v0, x)
    v2 = jnp.subtract(y, y)
    v3 = jnp.maximum(v2, v0)
    v4 = jnp.square(v1)
    return v4