import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.exp(x)
    v2 = jnp.add(x, y)
    v3 = jnp.tanh(v2)
    v4 = jnp.maximum(v2, v3)
    v5 = jnp.cos(x)
    return v5