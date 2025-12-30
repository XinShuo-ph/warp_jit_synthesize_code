import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.maximum(y, x)
    v2 = jnp.subtract(v1, v0)
    v3 = jnp.tanh(v0)
    v4 = jnp.abs(x)
    v5 = jnp.tanh(x)
    v6 = jnp.subtract(y, v4)
    v7 = jnp.exp(v4)
    v8 = jnp.minimum(y, v6)
    return v8