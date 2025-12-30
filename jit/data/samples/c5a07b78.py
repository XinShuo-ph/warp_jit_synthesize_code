import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.sin(y)
    v2 = jnp.tanh(y)
    return v2