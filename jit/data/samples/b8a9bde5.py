import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.abs(y)
    v2 = jnp.maximum(v1, y)
    v3 = jnp.exp(v2)
    v4 = jnp.cos(y)
    v5 = jnp.exp(v1)
    v6 = jnp.minimum(v3, y)
    v7 = jnp.maximum(v1, v3)
    v8 = jnp.cos(v2)
    v9 = jnp.tanh(y)
    return v9