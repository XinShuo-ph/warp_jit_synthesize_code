import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.add(x, v0)
    v2 = jnp.add(v1, y)
    return v2