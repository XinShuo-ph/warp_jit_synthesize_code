import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.cos(x)
    v2 = jnp.multiply(y, x)
    v3 = jnp.abs(y)
    v4 = jnp.minimum(v0, v3)
    return v4