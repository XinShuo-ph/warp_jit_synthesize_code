import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.maximum(v0, y)
    v2 = jnp.multiply(y, v0)
    v3 = jnp.multiply(v2, y)
    v4 = jnp.multiply(v3, v0)
    return v4