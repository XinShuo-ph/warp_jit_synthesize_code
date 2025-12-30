import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.tanh(x)
    v2 = jnp.exp(v1)
    v3 = jnp.tanh(x)
    v4 = jnp.cos(x)
    v5 = jnp.maximum(v2, v4)
    v6 = jnp.exp(x)
    v7 = jnp.minimum(v4, v4)
    v8 = jnp.minimum(v0, v4)
    return v8