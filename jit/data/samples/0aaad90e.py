import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.cos(x)
    v2 = jnp.multiply(v1, v0)
    v3 = jnp.subtract(v0, v1)
    v4 = jnp.square(y)
    v5 = jnp.multiply(x, v4)
    return v5