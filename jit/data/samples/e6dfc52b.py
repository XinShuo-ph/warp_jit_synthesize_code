import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.minimum(x, y)
    v2 = jnp.tanh(v0)
    v3 = jnp.multiply(v0, x)
    v4 = jnp.multiply(v3, v2)
    return v4