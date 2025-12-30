import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.subtract(y, v0)
    v2 = jnp.tanh(x)
    v3 = jnp.exp(v1)
    v4 = jnp.minimum(y, v1)
    v5 = jnp.abs(v1)
    v6 = jnp.square(v5)
    v7 = jnp.maximum(v1, v1)
    return v7