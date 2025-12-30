import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.maximum(x, y)
    v2 = jnp.exp(v0)
    return v2