import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.abs(x)
    v2 = jnp.maximum(x, y)
    v3 = jnp.add(y, v1)
    v4 = jnp.maximum(v2, v3)
    v5 = jnp.tanh(v0)
    v6 = jnp.add(v3, y)
    v7 = jnp.sin(v0)
    v8 = jnp.sin(v4)
    return v8