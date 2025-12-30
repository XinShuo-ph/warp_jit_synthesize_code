import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.abs(y)
    v2 = jnp.square(v1)
    v3 = jnp.abs(v0)
    return v3