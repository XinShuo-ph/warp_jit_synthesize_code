import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.add(v0, x)
    v2 = jnp.abs(v0)
    v3 = jnp.maximum(v1, v0)
    v4 = jnp.sin(v0)
    v5 = jnp.cos(y)
    v6 = jnp.sin(v3)
    return v6