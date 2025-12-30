import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, x)
    v1 = jnp.tanh(x)
    v2 = jnp.exp(v1)
    v3 = jnp.abs(x)
    v4 = jnp.tanh(y)
    v5 = jnp.subtract(y, v2)
    v6 = jnp.tanh(v2)
    v7 = jnp.tanh(v5)
    return v7