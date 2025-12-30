import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.multiply(v0, y)
    v2 = jnp.abs(v1)
    v3 = jnp.maximum(x, v0)
    v4 = jnp.subtract(y, x)
    v5 = jnp.square(v2)
    v6 = jnp.minimum(v4, x)
    return v6