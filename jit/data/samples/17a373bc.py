import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.abs(x)
    v2 = jnp.minimum(y, v0)
    v3 = jnp.cos(v1)
    v4 = jnp.square(v0)
    v5 = jnp.add(v3, v2)
    return v5