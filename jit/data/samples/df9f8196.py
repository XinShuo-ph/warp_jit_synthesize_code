import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.minimum(y, y)
    v2 = jnp.square(y)
    v3 = jnp.add(v0, v2)
    return v3