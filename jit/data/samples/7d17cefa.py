import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.sin(y)
    v2 = jnp.add(y, y)
    return v2