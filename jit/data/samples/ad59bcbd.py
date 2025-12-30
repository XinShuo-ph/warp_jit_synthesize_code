import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.tanh(x)
    v2 = jnp.maximum(y, v0)
    return v2