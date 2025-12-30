import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.cos(y)
    v2 = jnp.minimum(x, v0)
    v3 = jnp.tanh(v1)
    v4 = jnp.minimum(y, v0)
    v5 = jnp.abs(v1)
    v6 = jnp.maximum(x, v3)
    v7 = jnp.exp(v1)
    v8 = jnp.exp(y)
    v9 = jnp.abs(v5)
    return v9