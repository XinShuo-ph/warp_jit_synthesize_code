import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.subtract(v0, x)
    v2 = jnp.cos(y)
    return v2