import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.abs(y)
    v2 = jnp.subtract(x, v1)
    v3 = jnp.maximum(v0, v2)
    v4 = jnp.add(v3, v0)
    v5 = jnp.abs(v3)
    return v5