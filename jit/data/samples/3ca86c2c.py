import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.square(v0)
    v2 = jnp.maximum(v1, x)
    v3 = jnp.subtract(v0, x)
    return v3