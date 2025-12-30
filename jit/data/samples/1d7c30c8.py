import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.subtract(y, x)
    v2 = jnp.maximum(x, y)
    v3 = jnp.abs(v2)
    return v3