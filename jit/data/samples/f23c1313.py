import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(y)
    v1 = jnp.abs(v0)
    v2 = jnp.subtract(v1, v0)
    v3 = jnp.exp(v1)
    v4 = jnp.maximum(v3, v0)
    return v4