import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.tanh(v0)
    v2 = jnp.cos(v1)
    v3 = jnp.add(v0, v2)
    v4 = jnp.cos(y)
    v5 = jnp.abs(v4)
    return v5