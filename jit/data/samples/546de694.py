import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.square(v0)
    v3 = jnp.subtract(v2, v1)
    v4 = jnp.cos(v2)
    v5 = jnp.maximum(v4, v1)
    return v5