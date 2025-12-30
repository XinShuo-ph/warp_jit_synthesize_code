import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.subtract(x, y)
    v2 = jnp.multiply(v1, v1)
    return v2