import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.add(v0, y)
    v2 = jnp.multiply(v0, x)
    v3 = jnp.exp(v0)
    v4 = jnp.minimum(v2, x)
    v5 = jnp.sin(v3)
    v6 = jnp.multiply(v5, v5)
    v7 = jnp.subtract(v0, v1)
    return v7