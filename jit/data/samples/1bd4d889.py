import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.sin(v0)
    v2 = jnp.sin(v1)
    v3 = jnp.cos(v2)
    v4 = jnp.exp(v2)
    v5 = jnp.tanh(v1)
    v6 = jnp.maximum(v2, y)
    return v6