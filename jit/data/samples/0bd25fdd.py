import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.square(v0)
    v2 = jnp.square(x)
    v3 = jnp.subtract(y, v0)
    v4 = jnp.maximum(v1, v3)
    v5 = jnp.exp(y)
    v6 = jnp.cos(v0)
    v7 = jnp.multiply(v6, v0)
    return v7