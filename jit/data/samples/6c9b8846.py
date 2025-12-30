import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, x)
    v1 = jnp.add(v0, x)
    v2 = jnp.multiply(v1, y)
    v3 = jnp.square(v2)
    v4 = jnp.cos(v2)
    v5 = jnp.subtract(v2, v3)
    return v5