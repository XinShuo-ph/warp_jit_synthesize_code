import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.square(x)
    v2 = jnp.add(v1, v0)
    v3 = jnp.multiply(v1, v2)
    v4 = jnp.abs(v0)
    v5 = jnp.cos(v1)
    return v5