import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.abs(v0)
    v2 = jnp.subtract(v0, v0)
    v3 = jnp.add(x, v1)
    v4 = jnp.multiply(v3, v3)
    return v4