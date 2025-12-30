import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.add(x, v0)
    v2 = jnp.multiply(y, v0)
    v3 = jnp.tanh(y)
    v4 = jnp.add(v0, v1)
    return v4