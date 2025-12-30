import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.cos(x)
    v2 = jnp.tanh(v1)
    v3 = jnp.tanh(v2)
    v4 = jnp.tanh(v2)
    return v4