import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.maximum(y, y)
    v2 = jnp.maximum(v1, v1)
    v3 = jnp.tanh(v2)
    return v3