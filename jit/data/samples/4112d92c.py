import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.subtract(v0, v0)
    v2 = jnp.maximum(y, v0)
    return v2