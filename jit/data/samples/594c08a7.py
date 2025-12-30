import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, x)
    v1 = jnp.maximum(v0, y)
    v2 = jnp.exp(v0)
    v3 = jnp.cos(v1)
    v4 = jnp.square(v0)
    v5 = jnp.multiply(v0, v4)
    v6 = jnp.minimum(x, y)
    return v6