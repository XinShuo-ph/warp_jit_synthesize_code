import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.abs(x)
    v2 = jnp.subtract(y, v0)
    v3 = jnp.subtract(x, v0)
    return v3