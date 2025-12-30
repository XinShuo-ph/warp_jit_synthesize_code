import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, y)
    v1 = jnp.tanh(y)
    v2 = jnp.exp(x)
    v3 = jnp.exp(y)
    return v3