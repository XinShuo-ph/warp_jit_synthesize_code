import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.add(y, x)
    v2 = jnp.subtract(y, x)
    v3 = jnp.abs(v1)
    v4 = jnp.cos(v0)
    v5 = jnp.cos(y)
    return v5