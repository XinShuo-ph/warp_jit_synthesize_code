import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.square(v0)
    v2 = jnp.add(v0, x)
    v3 = jnp.square(v1)
    v4 = jnp.abs(v3)
    v5 = jnp.square(v2)
    v6 = jnp.subtract(v5, v2)
    return v6