import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.cos(y)
    v2 = jnp.minimum(v1, v0)
    return v2