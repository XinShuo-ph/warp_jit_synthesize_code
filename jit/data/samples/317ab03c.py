import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.sin(v0)
    v2 = jnp.add(v1, x)
    return v2