import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.cos(y)
    v2 = jnp.minimum(v0, x)
    v3 = jnp.cos(v1)
    v4 = jnp.abs(v3)
    v5 = jnp.minimum(v0, v4)
    v6 = jnp.square(v0)
    return v6