import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.cos(y)
    v2 = jnp.minimum(x, x)
    v3 = jnp.multiply(v1, v1)
    v4 = jnp.multiply(v1, v1)
    v5 = jnp.minimum(y, v4)
    return v5