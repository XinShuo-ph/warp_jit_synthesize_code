import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.minimum(y, x)
    v2 = jnp.tanh(y)
    v3 = jnp.sin(v2)
    return v3