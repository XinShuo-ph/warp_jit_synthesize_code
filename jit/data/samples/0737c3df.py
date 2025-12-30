import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.multiply(y, y)
    v2 = jnp.square(v1)
    return v2