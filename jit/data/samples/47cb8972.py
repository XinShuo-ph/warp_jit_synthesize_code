import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.tanh(y)
    v2 = jnp.minimum(y, y)
    v3 = jnp.square(v2)
    v4 = jnp.tanh(v1)
    v5 = jnp.cos(x)
    v6 = jnp.maximum(v3, v4)
    v7 = jnp.multiply(v0, v5)
    return v7