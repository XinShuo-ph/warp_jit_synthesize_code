import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.exp(y)
    v2 = jnp.maximum(v1, v0)
    v3 = jnp.square(v0)
    v4 = jnp.multiply(v1, v3)
    return v4