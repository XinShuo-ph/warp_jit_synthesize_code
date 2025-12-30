import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.subtract(y, v0)
    v2 = jnp.add(v0, v1)
    v3 = jnp.cos(v1)
    v4 = jnp.exp(v2)
    v5 = jnp.subtract(v2, x)
    return v5