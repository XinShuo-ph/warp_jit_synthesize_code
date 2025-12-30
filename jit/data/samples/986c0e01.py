import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.add(x, v0)
    v2 = jnp.maximum(x, y)
    v3 = jnp.cos(v2)
    v4 = jnp.tanh(v1)
    v5 = jnp.maximum(v1, v2)
    v6 = jnp.add(y, v3)
    v7 = jnp.subtract(v3, v2)
    v8 = jnp.add(v4, v7)
    v9 = jnp.cos(v4)
    return v9