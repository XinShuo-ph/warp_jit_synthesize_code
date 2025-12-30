import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.add(y, x)
    v2 = jnp.exp(v1)
    v3 = jnp.cos(v0)
    v4 = jnp.abs(v2)
    v5 = jnp.maximum(v3, v4)
    v6 = jnp.subtract(v1, v3)
    return v6