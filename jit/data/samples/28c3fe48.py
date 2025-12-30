import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.cos(x)
    v2 = jnp.cos(y)
    v3 = jnp.tanh(x)
    v4 = jnp.tanh(v3)
    return v4