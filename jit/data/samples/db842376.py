import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.cos(v0)
    v2 = jnp.tanh(y)
    v3 = jnp.maximum(v0, v2)
    return v3