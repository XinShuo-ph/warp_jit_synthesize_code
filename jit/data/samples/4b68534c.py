import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, y)
    v1 = jnp.tanh(v0)
    v2 = jnp.abs(v0)
    v3 = jnp.add(v2, v2)
    v4 = jnp.tanh(v3)
    v5 = jnp.cos(y)
    v6 = jnp.maximum(v5, v2)
    return v6