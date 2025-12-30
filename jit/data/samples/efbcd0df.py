import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.abs(v0)
    v2 = jnp.subtract(y, y)
    v3 = jnp.square(x)
    v4 = jnp.abs(v2)
    v5 = jnp.minimum(v3, v1)
    return v5