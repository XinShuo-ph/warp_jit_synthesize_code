import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.abs(y)
    v2 = jnp.cos(y)
    v3 = jnp.minimum(x, v0)
    v4 = jnp.add(v0, x)
    v5 = jnp.square(v2)
    v6 = jnp.cos(v3)
    return v6