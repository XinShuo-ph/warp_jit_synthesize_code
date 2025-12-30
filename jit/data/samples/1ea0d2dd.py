import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.sin(y)
    v2 = jnp.square(y)
    v3 = jnp.abs(v1)
    v4 = jnp.add(v1, v1)
    v5 = jnp.cos(y)
    v6 = jnp.multiply(y, v4)
    v7 = jnp.square(v6)
    v8 = jnp.exp(v0)
    return v8