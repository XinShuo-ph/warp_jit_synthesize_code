import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.cos(y)
    v2 = jnp.sin(y)
    v3 = jnp.cos(y)
    v4 = jnp.sin(x)
    v5 = jnp.tanh(v1)
    return v5