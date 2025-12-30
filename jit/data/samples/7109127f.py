import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.abs(x)
    v2 = jnp.square(v0)
    v3 = jnp.subtract(v0, y)
    v4 = jnp.minimum(v2, x)
    v5 = jnp.cos(x)
    v6 = jnp.minimum(x, v3)
    return v6