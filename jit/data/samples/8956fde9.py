import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.tanh(y)
    v2 = jnp.square(v1)
    v3 = jnp.multiply(y, v1)
    v4 = jnp.maximum(y, v0)
    return v4