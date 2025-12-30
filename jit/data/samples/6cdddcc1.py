import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.multiply(y, v0)
    v2 = jnp.abs(v1)
    v3 = jnp.add(x, x)
    v4 = jnp.sin(v1)
    v5 = jnp.subtract(v2, v4)
    v6 = jnp.abs(v1)
    v7 = jnp.subtract(v0, v0)
    return v7