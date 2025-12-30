import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.abs(y)
    v2 = jnp.minimum(v0, x)
    v3 = jnp.minimum(v1, y)
    v4 = jnp.subtract(v3, x)
    v5 = jnp.square(y)
    return v5