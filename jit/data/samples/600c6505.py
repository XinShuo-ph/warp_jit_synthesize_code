import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.subtract(x, x)
    v2 = jnp.abs(x)
    v3 = jnp.sin(v1)
    v4 = jnp.cos(v0)
    v5 = jnp.sin(x)
    v6 = jnp.add(v0, v2)
    v7 = jnp.multiply(v2, v2)
    v8 = jnp.add(v7, v7)
    v9 = jnp.multiply(v2, v5)
    return v9