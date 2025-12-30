import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.square(x)
    v2 = jnp.cos(v1)
    return v2