import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.cos(y)
    v2 = jnp.tanh(v1)
    v3 = jnp.tanh(v1)
    v4 = jnp.minimum(v1, v2)
    v5 = jnp.multiply(v0, v3)
    return v5