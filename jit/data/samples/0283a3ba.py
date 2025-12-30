import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.sin(v0)
    v2 = jnp.cos(x)
    v3 = jnp.minimum(y, v2)
    return v3