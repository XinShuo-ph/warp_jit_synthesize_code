import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.maximum(y, v0)
    v2 = jnp.maximum(v1, v1)
    v3 = jnp.maximum(v2, x)
    v4 = jnp.sin(v0)
    v5 = jnp.multiply(v4, v3)
    v6 = jnp.sin(v3)
    v7 = jnp.tanh(v6)
    v8 = jnp.minimum(v2, v6)
    return v8