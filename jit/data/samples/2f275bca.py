import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.tanh(v0)
    v2 = jnp.cos(x)
    v3 = jnp.maximum(y, y)
    v4 = jnp.tanh(v0)
    v5 = jnp.sin(v3)
    v6 = jnp.exp(v3)
    return v6