import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.square(x)
    v2 = jnp.maximum(y, x)
    v3 = jnp.add(v1, x)
    v4 = jnp.add(y, v1)
    v5 = jnp.abs(y)
    v6 = jnp.cos(v1)
    return v6