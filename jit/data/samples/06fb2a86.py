import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.tanh(x)
    v2 = jnp.sin(v1)
    v3 = jnp.sin(y)
    v4 = jnp.maximum(y, v1)
    v5 = jnp.add(v4, x)
    v6 = jnp.abs(v4)
    return v6