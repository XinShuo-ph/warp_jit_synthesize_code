import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.sin(x)
    v2 = jnp.add(y, v1)
    v3 = jnp.multiply(y, v0)
    v4 = jnp.cos(v1)
    v5 = jnp.abs(x)
    v6 = jnp.minimum(y, y)
    v7 = jnp.abs(y)
    v8 = jnp.subtract(v5, v6)
    return v8