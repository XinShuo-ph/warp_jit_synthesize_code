import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.maximum(v0, v0)
    v3 = jnp.tanh(v1)
    v4 = jnp.square(y)
    return v4