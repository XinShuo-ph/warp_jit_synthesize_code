import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.sin(x)
    v2 = jnp.tanh(v0)
    v3 = jnp.sin(v1)
    v4 = jnp.exp(x)
    v5 = jnp.maximum(v2, v0)
    return v5