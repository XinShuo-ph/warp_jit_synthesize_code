import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.cos(x)
    v2 = jnp.tanh(y)
    return v2