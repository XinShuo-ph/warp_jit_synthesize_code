import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.sin(y)
    v2 = jnp.cos(y)
    v3 = jnp.tanh(v0)
    v4 = jnp.tanh(v0)
    v5 = jnp.maximum(v0, y)
    return v5