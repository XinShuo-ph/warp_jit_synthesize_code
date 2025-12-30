import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, x)
    v1 = jnp.subtract(x, y)
    v2 = jnp.abs(v1)
    v3 = jnp.abs(x)
    v4 = jnp.add(x, y)
    v5 = jnp.minimum(x, v1)
    return v5