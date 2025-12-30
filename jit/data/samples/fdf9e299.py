import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.sin(v0)
    v2 = jnp.multiply(y, v1)
    return v2