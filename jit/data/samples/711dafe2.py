import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.cos(v0)
    v2 = jnp.sin(v0)
    v3 = jnp.maximum(y, v1)
    return v3