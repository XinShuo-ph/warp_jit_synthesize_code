import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.exp(x)
    v2 = jnp.maximum(y, v0)
    v3 = jnp.sin(x)
    v4 = jnp.abs(v3)
    v5 = jnp.tanh(v4)
    v6 = jnp.exp(v2)
    return v6