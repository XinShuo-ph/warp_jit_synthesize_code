import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.abs(x)
    v2 = jnp.minimum(v1, x)
    v3 = jnp.cos(x)
    v4 = jnp.square(v2)
    v5 = jnp.minimum(v2, v4)
    return v5