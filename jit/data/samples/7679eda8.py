import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(y)
    v1 = jnp.exp(y)
    v2 = jnp.tanh(v0)
    v3 = jnp.maximum(v0, v1)
    return v3