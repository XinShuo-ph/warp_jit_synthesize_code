import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.cos(x)
    v2 = jnp.maximum(y, v1)
    return v2