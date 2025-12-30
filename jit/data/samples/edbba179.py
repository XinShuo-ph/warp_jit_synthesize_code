import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.abs(y)
    v2 = jnp.sin(x)
    v3 = jnp.multiply(v0, y)
    v4 = jnp.tanh(v0)
    v5 = jnp.tanh(v3)
    v6 = jnp.maximum(v1, x)
    return v6