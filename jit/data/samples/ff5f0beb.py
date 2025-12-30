import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.maximum(y, x)
    v2 = jnp.maximum(v1, v0)
    v3 = jnp.tanh(v2)
    v4 = jnp.add(v3, x)
    v5 = jnp.tanh(v2)
    v6 = jnp.tanh(v1)
    return v6