import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.tanh(v0)
    v2 = jnp.minimum(v0, y)
    return v2