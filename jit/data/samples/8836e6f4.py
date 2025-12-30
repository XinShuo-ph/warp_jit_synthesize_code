import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.square(y)
    v2 = jnp.square(y)
    v3 = jnp.multiply(v1, y)
    return v3