import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.minimum(x, y)
    v2 = jnp.multiply(v0, v1)
    v3 = jnp.square(x)
    return v3