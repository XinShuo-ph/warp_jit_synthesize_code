import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.tanh(y)
    v2 = jnp.cos(y)
    v3 = jnp.cos(v0)
    v4 = jnp.square(v2)
    return v4