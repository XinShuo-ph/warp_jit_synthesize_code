import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, x)
    v1 = jnp.multiply(v0, y)
    v2 = jnp.subtract(v1, x)
    v3 = jnp.minimum(y, y)
    v4 = jnp.add(x, v1)
    return v4