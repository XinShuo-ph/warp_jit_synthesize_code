import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.abs(y)
    v2 = jnp.minimum(v1, v1)
    v3 = jnp.tanh(x)
    v4 = jnp.add(y, y)
    v5 = jnp.tanh(v3)
    v6 = jnp.tanh(x)
    v7 = jnp.subtract(v2, y)
    v8 = jnp.subtract(v1, y)
    return v8