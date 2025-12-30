import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.multiply(y, v0)
    v2 = jnp.multiply(y, y)
    v3 = jnp.abs(v1)
    return v3