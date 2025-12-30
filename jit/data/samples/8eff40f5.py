import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.minimum(v0, x)
    v2 = jnp.add(x, y)
    return v2