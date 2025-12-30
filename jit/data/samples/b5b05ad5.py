import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.cos(y)
    v2 = jnp.subtract(x, x)
    v3 = jnp.square(x)
    v4 = jnp.subtract(y, v3)
    v5 = jnp.subtract(v1, v4)
    v6 = jnp.abs(v3)
    v7 = jnp.add(v5, v2)
    v8 = jnp.maximum(v7, v0)
    v9 = jnp.cos(v1)
    return v9