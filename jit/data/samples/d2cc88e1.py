import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, y)
    v1 = jnp.cos(x)
    v2 = jnp.square(v1)
    v3 = jnp.minimum(v0, v2)
    return v3