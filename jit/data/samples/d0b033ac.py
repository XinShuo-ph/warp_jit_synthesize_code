import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(y)
    v1 = jnp.abs(x)
    v2 = jnp.minimum(v0, x)
    v3 = jnp.maximum(y, x)
    v4 = jnp.multiply(y, v1)
    return v4