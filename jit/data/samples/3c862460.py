import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.tanh(x)
    v2 = jnp.maximum(x, v1)
    v3 = jnp.maximum(v2, v0)
    v4 = jnp.abs(v0)
    v5 = jnp.multiply(v3, v2)
    v6 = jnp.tanh(y)
    v7 = jnp.multiply(v6, y)
    v8 = jnp.cos(v3)
    v9 = jnp.square(v5)
    return v9