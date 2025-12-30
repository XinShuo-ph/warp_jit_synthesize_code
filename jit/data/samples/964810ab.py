import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.maximum(v0, x)
    v2 = jnp.add(x, v0)
    v3 = jnp.maximum(v0, v0)
    v4 = jnp.add(v0, v2)
    v5 = jnp.sin(x)
    v6 = jnp.maximum(v3, x)
    v7 = jnp.exp(v3)
    return v7