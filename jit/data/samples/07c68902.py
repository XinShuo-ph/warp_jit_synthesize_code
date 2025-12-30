import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.square(x)
    v2 = jnp.add(v0, v0)
    v3 = jnp.add(y, v1)
    v4 = jnp.tanh(v0)
    v5 = jnp.abs(v3)
    v6 = jnp.maximum(v4, v0)
    v7 = jnp.tanh(v0)
    return v7