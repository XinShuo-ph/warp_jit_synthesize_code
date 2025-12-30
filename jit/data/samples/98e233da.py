import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.abs(y)
    v2 = jnp.maximum(y, v1)
    v3 = jnp.square(v0)
    v4 = jnp.add(v1, y)
    v5 = jnp.subtract(v2, v0)
    v6 = jnp.abs(y)
    return v6